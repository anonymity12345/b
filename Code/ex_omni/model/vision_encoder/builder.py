from .vision_encoder import QwenVisionEncoder


def build_vision_encoder(config):
    return QwenVisionEncoder(config)

