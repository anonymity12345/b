(() => {
    // Gradio transports URLs only. It never owns or replaces these media elements.
    let player;
    const bind = () => {
        const stage = document.getElementById('avatar-stage');
        if (!stage) return;
        if (!player || player.stage !== stage) player = createPlayer(stage);
        const url = id => document.querySelector(`#${id} [data-video-src]`)?.dataset.videoSrc || '';
        const complete = document.querySelector('#avatar-reply-source [data-video-src]')?.dataset.videoComplete === '1';
        player.update(url('avatar-idle-source'), url('avatar-reply-source'), complete);
    };
    function createPlayer(stage) {
        const idle = stage.querySelector('#avatar-idle-video');
        const reply = stage.querySelector('#multimodal-response-video');
        const frame = stage.querySelector('canvas');
        const button = stage.querySelector('button');
        let idleURL = '', replyURL = '', blocked = '', cursor = 0, timer;
        let speaking = false, complete = false, disposeStream;
        const closeStream = () => { disposeStream?.(); disposeStream = undefined; };
        const startFragments = (manifestURL) => {
            const controller = new AbortController();
            let objectURL, stopped = false;
            const stats = {segments: 0, bytes: 0, sourceChanges: 0, started: performance.now()};
            window.exOmniMediaStats = stats;
            const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
            const fetchData = async (url, binary = false) => {
                for (let attempt = 0; ; attempt++) {
                    try {
                        const response = await fetch(url, {signal: controller.signal, cache: 'no-store'});
                        if (!response.ok) throw new Error(`Media HTTP ${response.status}`);
                        return binary ? await response.arrayBuffer() : await response.json();
                    } catch (error) {
                        if (stopped || error.name === 'AbortError' || attempt >= 5) throw error;
                        await delay(200 * (attempt + 1));
                    }
                }
            };
            const once = (target, event) => new Promise((resolve, reject) => {
                const clean = () => {
                    target.removeEventListener(event, done);
                    target.removeEventListener('error', fail);
                    controller.signal.removeEventListener('abort', fail);
                };
                const done = () => { clean(); resolve(); };
                const fail = () => { clean(); reject(new Error('Media append interrupted')); };
                target.addEventListener(event, done, {once: true});
                target.addEventListener('error', fail, {once: true});
                controller.signal.addEventListener('abort', fail, {once: true});
            });
            const fileURL = path => '/gradio_api/file=' + encodeURI(path).replaceAll('#', '%23').replaceAll('?', '%3F');
            const fallback = path => {
                if (stopped || !path) return;
                cursor = Math.max(cursor, reply.currentTime || 0);
                reply.src = fileURL(path); stats.sourceChanges++; reply.load();
            };
            (async () => {
                let manifest = await fetchData(manifestURL);
                const Source = window.MediaSource || window.ManagedMediaSource;
                if (!Source || !Source.isTypeSupported(manifest.mime)) {
                    while (!manifest.complete && !manifest.error && !stopped) {
                        await delay(250); manifest = await fetchData(manifestURL);
                    }
                    fallback(manifest.fallback); return;
                }
                const media = new Source();
                reply.disableRemotePlayback = true;
                objectURL = URL.createObjectURL(media);
                const opened = once(media, 'sourceopen');
                reply.src = objectURL; stats.sourceChanges++; reply.load();
                await opened;
                const buffer = media.addSourceBuffer(manifest.mime);
                const append = async data => {
                    if (stopped) return;
                    const updated = once(buffer, 'updateend');
                    buffer.appendBuffer(data); await updated;
                };
                const base = manifestURL.slice(0, manifestURL.lastIndexOf('/') + 1);
                let initialized = false, index = 0;
                const downloads = new Map();
                const prefetch = () => {
                    for (let next = index; next < Math.min(index + 3, manifest.segments.length); next++) {
                        if (!downloads.has(next)) {
                            const promise = fetchData(base + manifest.segments[next], true);
                            promise.catch(() => {}); // awaited in order below
                            downloads.set(next, promise);
                        }
                    }
                };
                while (!stopped) {
                    if (manifest.error) throw new Error(manifest.error);
                    if (manifest.segments.length && !initialized) {
                        const data = await fetchData(base + manifest.init, true);
                        stats.bytes += data.byteLength; await append(data); initialized = true;
                    }
                    while (index < manifest.segments.length && !stopped) {
                        prefetch();
                        const data = await downloads.get(index);
                        downloads.delete(index);
                        stats.bytes += data.byteLength; await append(data);
                        index++; stats.segments++;
                        stats.firstAppend ??= performance.now();
                        if (reply.paused && !blocked) play();
                    }
                    if (manifest.complete) {
                        if (media.readyState === 'open') media.endOfStream();
                        stats.complete = performance.now(); break;
                    }
                    await delay(120); manifest = await fetchData(manifestURL);
                }
            })().catch(async error => {
                if (stopped || error.name === 'AbortError') return;
                stats.error = String(error);
                // A complete MP4 remains available for browsers without MSE,
                // or for a recoverable media decode error.
                try {
                    let manifest = await fetchData(manifestURL);
                    while (!manifest.complete && !manifest.error && !stopped) {
                        await delay(250); manifest = await fetchData(manifestURL);
                    }
                    fallback(manifest.fallback);
                } catch (_) { if (!stopped) button.hidden = false; }
            });
            return () => {
                stopped = true; controller.abort();
                if (objectURL) URL.revokeObjectURL(objectURL);
            };
        };
        const capture = () => {
            if (reply.readyState < 2 || !reply.videoWidth) return;
            frame.width = reply.videoWidth; frame.height = reply.videoHeight;
            frame.getContext('2d').drawImage(reply, 0, 0);
        };
        const standby = () => {
            clearTimeout(timer); speaking = false;
            reply.pause(); reply.hidden = true; frame.hidden = true;
            idle.hidden = false; button.hidden = true;
            stage.classList.remove('reply-playing');
            if (idleURL) idle.play().catch(() => {});
        };
        const hold = () => {
            if (!speaking) return;
            capture(); frame.hidden = false; reply.hidden = true;
        };
        const play = () => {
            if (!replyURL || blocked === replyURL) return;
            const requested = replyURL;
            reply.play().catch(error => {
                if (requested === replyURL && error.name !== 'AbortError') button.hidden = false;
            });
        };
        const reset = () => {
            closeStream(); blocked = replyURL; cursor = 0; standby();
        };
        window.exOmniResetReply = reset;
        idle.muted = true; idle.defaultMuted = true; idle.loop = true;
        idle.addEventListener('canplay', () => { if (!speaking) idle.play().catch(() => {}); });
        idle.addEventListener('play', () => { if (speaking) idle.pause(); });
        reply.addEventListener('loadedmetadata', () => {
            if (!replyURL || blocked === replyURL) return;
            if (cursor >= reply.duration - .06) {
                if (complete) timer = setTimeout(standby, 650);
                return;
            }
            if (cursor > 0) reply.currentTime = cursor;
            play();
        });
        reply.addEventListener('playing', () => {
            if (!replyURL || blocked === replyURL) { reply.pause(); return; }
            clearTimeout(timer); speaking = true;
            idle.pause(); idle.hidden = true;
            reply.hidden = false; frame.hidden = true; button.hidden = true;
            stage.classList.add('reply-playing');
        });
        reply.addEventListener('timeupdate', () => {
            if (!blocked && replyURL) cursor = Math.max(cursor, reply.currentTime);
        });
        reply.addEventListener('waiting', hold);
        reply.addEventListener('ended', () => {
            cursor = Math.max(cursor, reply.currentTime); hold();
            clearTimeout(timer);
            if (complete) timer = setTimeout(standby, 650);
        });
        reply.addEventListener('error', () => { if (replyURL) button.hidden = false; });
        button.onclick = () => { blocked = ''; play(); };
        return { stage, update(nextIdle, nextReply, nextComplete) {
            if (nextComplete !== complete) {
                complete = nextComplete;
                if (complete && reply.ended) timer = setTimeout(standby, 650);
            }
            if (nextIdle !== idleURL) {
                idleURL = nextIdle;
                if (idleURL) { idle.src = idleURL; if (!speaking) idle.play().catch(() => {}); }
                else { idle.pause(); idle.removeAttribute('src'); idle.load(); }
            }
            if (nextReply === replyURL) return;
            clearTimeout(timer);
            if (!nextReply) {
                closeStream(); replyURL = ''; blocked = ''; cursor = 0; standby();
                reply.removeAttribute('src'); reply.load(); return;
            }
            hold(); reply.pause();
            closeStream(); replyURL = nextReply; blocked = '';
            if (replyURL.endsWith('/stream.json')) disposeStream = startFragments(replyURL);
            else { reply.src = replyURL; reply.load(); }
        }};
    }
    bind();
    new MutationObserver(bind).observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['data-video-src', 'data-video-complete']});
})();
