"""Encode each frame once into immutable H.264/AAC fMP4 fragments."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import numpy as np
from ex_omni.media import ffmpeg_executable


class FragmentedMediaRecorder:
    def __init__(self, output_path, *, fps, ffmpeg_path=None):
        self.output_path = Path(output_path).resolve()
        self.directory = self.output_path.parent / (self.output_path.stem + '-stream')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / 'stream.json'
        self.fps = float(fps)
        self.ffmpeg = ffmpeg_executable(ffmpeg_path)
        self.frames = 0
        self.finished = False
        self._samples = 0
        self._process = None
        self._errors = []
        self._queues = []
        self._writers = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._complete = False
        self._fallback = None
        self._last_manifest = None
        self._publish()

    def _publish(self):
        with self._lock:
            payload = dict(version=1, mime='video/mp4; codecs="avc1.64001f, mp4a.40.2"',
                           init='init.mp4', segments=[p.name for p in sorted(self.directory.glob('segment-*.m4s'))],
                           complete=self._complete, fallback=str(self._fallback) if self._fallback else None,
                           error=str(self._errors[0]) if self._errors else None)
            text = json.dumps(payload)
            if text != self._last_manifest:
                temporary = self.manifest_path.with_suffix('.tmp')
                temporary.write_text(text)
                temporary.replace(self.manifest_path)
                self._last_manifest = text

    def _watch(self):
        while not self._stop.wait(.05):
            self._publish()

    def _write(self, fd, items):
        try:
            with os.fdopen(fd, 'wb', buffering=0) as stream:
                while True:
                    item = items.get()
                    if item is None:
                        break
                    view = memoryview(item)
                    while view:
                        written = stream.write(view)
                        if not written:
                            raise BrokenPipeError('media encoder input closed')
                        view = view[written:]
        except Exception as exc:
            self._errors.append(exc)

    def _open(self, frames, sample_rate):
        self._sample_rate = int(sample_rate)
        height, width = frames.shape[1:3]
        self._shape = (height, width)
        pipes = [os.pipe(), os.pipe()]
        reads = [pair[0] for pair in pipes]
        gop = max(1, round(self.fps * .48))
        command = [self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                   '-thread_queue_size', '32', '-probesize', '32', '-analyzeduration', '0',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                   '-framerate', str(self.fps), '-i', f'pipe:{reads[0]}',
                   '-thread_queue_size', '32', '-probesize', '32', '-analyzeduration', '0',
                   '-f', 'f32le', '-ar', str(sample_rate), '-ac', '1', '-i', f'pipe:{reads[1]}',
                   '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-threads', '2',
                   '-preset', 'veryfast', '-tune', 'zerolatency', '-crf', '23',
                   '-profile:v', 'high', '-level:v', '3.1', '-pix_fmt', 'yuv420p',
                   '-g', str(gop), '-keyint_min', str(gop), '-sc_threshold', '0', '-bf', '0',
                   '-c:a', 'aac', '-b:a', '96k', '-flush_packets', '1',
                   '-f', 'hls', '-hls_time', str(gop / self.fps), '-hls_list_size', '0',
                   '-hls_segment_type', 'fmp4', '-hls_flags', 'independent_segments+temp_file',
                   '-hls_fmp4_init_filename', 'init.mp4',
                   '-hls_segment_filename', str(self.directory / 'segment-%06d.m4s'),
                   str(self.directory / 'index.m3u8')]
        self._log = (self.directory / 'encoder.log').open('wb')
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=self._log, pass_fds=reads)
        except Exception:
            self._log.close()
            for pair in pipes:
                for fd in pair:
                    os.close(fd)
            raise
        for read, write in pipes:
            os.close(read)
            items = queue.Queue(maxsize=8)
            worker = threading.Thread(target=self._write, args=(write, items), daemon=True)
            worker.start()
            self._queues.append(items)
            self._writers.append(worker)
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()

    def append(self, event):
        if self.finished or self._stop.is_set():
            raise RuntimeError('media stream is closed')
        frames = np.ascontiguousarray(event.frames, dtype=np.uint8)
        if int(event.start_frame) != self.frames:
            raise ValueError('non-contiguous media frames')
        if self._process is None:
            self._open(frames, int(event.audio_sample_rate))
        if tuple(frames.shape[1:3]) != self._shape or int(event.audio_sample_rate) != self._sample_rate:
            raise ValueError('media format changed during streaming')
        if self._errors or self._process.poll() is not None:
            raise RuntimeError(f'media encoder failed: {self._errors}; see {self.directory / "encoder.log"}')
        waveform = np.asarray(event.waveform, dtype='<f4').reshape(-1)
        start = int(event.audio_start_sample)
        if start > self._samples:
            waveform = np.concatenate([np.zeros(start-self._samples, dtype='<f4'), waveform])
        elif start < self._samples:
            waveform = waveform[min(self._samples-start, len(waveform)):]
        self.frames += len(frames)
        target = round(self.frames / self.fps * self._sample_rate)
        waveform = waveform[:max(0, target-self._samples)]
        self._samples += len(waveform)
        self._queues[0].put(frames.tobytes(), timeout=30)
        self._queues[1].put(waveform.tobytes(), timeout=30)

    def finalize(self):
        if self.finished:
            return self.output_path
        if self._process is None:
            raise RuntimeError('media stream contains no frames')
        target = round(self.frames / self.fps * self._sample_rate)
        if self._samples < target:
            self._queues[1].put(np.zeros(target-self._samples, dtype='<f4').tobytes(), timeout=30)
        for items in self._queues:
            items.put(None, timeout=30)
        for worker in self._writers:
            worker.join(timeout=30)
            if worker.is_alive():
                raise TimeoutError('media writer did not finish')
        code = self._process.wait(timeout=60)
        self._log.close()
        if code or self._errors:
            raise RuntimeError(f'media encoder failed: exit={code}, errors={self._errors}')
        # Same encoded packets for the downloadable final MP4; no second encode.
        subprocess.run([self.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                        '-allowed_extensions', 'ALL', '-i', str(self.directory / 'index.m3u8'),
                        '-c', 'copy', '-movflags', '+faststart',
                        str(self.output_path)], check=True, timeout=60)
        self._fallback = self.output_path
        self._complete = True
        self.finished = True
        self._stop.set()
        self._watcher.join(timeout=2)
        self._publish()
        return self.output_path

    def abort(self):
        if self.finished:
            return
        self._errors.append(RuntimeError('media stream aborted'))
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            for items in self._queues:
                while True:
                    try:items.get_nowait()
                    except queue.Empty:break
                items.put_nowait(None)
            for worker in self._writers:
                worker.join(timeout=2)
            self._log.close()
        self._publish()
