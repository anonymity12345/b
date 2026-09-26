import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from safetensors import safe_open


@dataclass
class Qwen3TTSSpeakerEncoderConfig:
    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    enc_kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000

    @classmethod
    def from_pretrained_config(cls, model_path: str):
        config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
        values = {}
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                values = json.load(f).get("speaker_encoder_config", {})
        return cls(**values)


class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        output_part = None
        for idx, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if idx == 0:
                output_part = hidden_part
            elif idx == 1:
                output_part = self.blocks[idx - 1](hidden_part)
            else:
                output_part = self.blocks[idx - 1](hidden_part + output_part)
            outputs.append(output_part)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)
        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))
        return hidden_states * hidden_states_mean


class SqueezeExcitationRes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.tdnn1(hidden_states)
        hidden_states = self.res2net_block(hidden_states)
        hidden_states = self.tdnn2(hidden_states)
        hidden_states = self.se_block(hidden_states)
        return hidden_states + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        if max_len is None:
            max_len = length.max().long().item()
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len)
        mask = mask < length.unsqueeze(1)
        return torch.as_tensor(mask, dtype=dtype, device=device)

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)
        mask = self._length_to_mask(
            lengths * seq_length,
            max_len=seq_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).unsqueeze(1)
        total = mask.sum(dim=2, keepdim=True)
        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)
        attention = self.conv(self.tanh(self.tdnn(attention)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        return torch.cat((mean, std), dim=1).unsqueeze(2)


class Qwen3TTSSpeakerEncoder(nn.Module):
    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(config.enc_dilations):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have the same length")

        self.blocks = nn.ModuleList()
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )
        for idx in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[idx - 1],
                    config.enc_channels[idx],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[idx],
                    dilation=config.enc_dilations[idx],
                )
            )
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)
        hidden_states = self.asp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states.squeeze(-1)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    if y.shape[-1] <= n_fft:
        y = F.pad(y, (0, n_fft + 1 - y.shape[-1]))
    device = y.device
    mel_basis = AF.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=float(fmin),
        f_max=float(fmax or sampling_rate // 2),
        n_mels=num_mels,
        sample_rate=sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    ).transpose(0, 1).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)
    padding = (n_fft - hop_size) // 2
    y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    return dynamic_range_compression_torch(mel_spec)


def speaker_mel_on_cpu_by_default():
    value = os.environ.get("EXOMNI_SPEAKER_MEL_ON_CPU", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _resolve_qwen3_tts_safetensors(model_path: str) -> str:
    if not model_path or str(model_path).lower() == "none":
        raise ValueError("pretrain_speaker_encoder_weights must point to Qwen3-TTS model weights.")
    model_path = os.path.expanduser(model_path)
    candidates = [model_path] if os.path.isfile(model_path) else [os.path.join(model_path, "model.safetensors")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find Qwen3-TTS model.safetensors under {model_path}")


class Qwen3TTSSpeakerEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.config = Qwen3TTSSpeakerEncoderConfig.from_pretrained_config(model_path)
        self.encoder = Qwen3TTSSpeakerEncoder(self.config)
        self._load_weights(model_path)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self):
        return int(self.config.enc_dim)

    @property
    def sample_rate(self):
        return int(self.config.sample_rate)

    def _load_weights(self, model_path: str):
        safetensors_path = _resolve_qwen3_tts_safetensors(model_path)
        state_dict = {}
        prefix = "speaker_encoder."
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state_dict[key[len(prefix):]] = handle.get_tensor(key)
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Failed to load Qwen3-TTS speaker encoder. missing={missing}, unexpected={unexpected}")

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor | None = None) -> torch.Tensor:
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveform_lengths is None:
            waveform_lengths = torch.full(
                (waveforms.shape[0],),
                waveforms.shape[-1],
                dtype=torch.long,
                device=waveforms.device,
            )
        waveform_lengths = waveform_lengths.view(-1).to(device=waveforms.device)

        outputs = torch.zeros(
            waveforms.shape[0],
            self.embedding_dim,
            dtype=torch.float32,
            device=waveforms.device,
        )
        for idx in range(waveforms.shape[0]):
            length = int(waveform_lengths[idx].item())
            if length <= 0:
                continue
            wav = waveforms[idx, :length].float().unsqueeze(0).clamp(min=-1.0, max=1.0)
            mel_input = wav.cpu() if speaker_mel_on_cpu_by_default() else wav
            try:
                mel = mel_spectrogram(
                    mel_input,
                    n_fft=1024,
                    num_mels=self.config.mel_dim,
                    sampling_rate=self.config.sample_rate,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=self.config.sample_rate // 2,
                    center=False,
                )
            except RuntimeError as exc:
                if not mel_input.is_cuda or "cuFFT" not in str(exc):
                    raise
                mel = mel_spectrogram(
                    wav.cpu(),
                    n_fft=1024,
                    num_mels=self.config.mel_dim,
                    sampling_rate=self.config.sample_rate,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=self.config.sample_rate // 2,
                    center=False,
                )
            encoder_dtype = next(self.encoder.parameters()).dtype
            mel = mel.transpose(1, 2).to(device=waveforms.device, dtype=encoder_dtype)
            outputs[idx] = self.encoder(mel).squeeze(0)
        return outputs
