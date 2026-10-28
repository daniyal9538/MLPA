"""MLPA: Missing-modality-aware Latent Prediction Architecture (multimodal bottleneck fusion of frozen encoders)."""
from .dropout import sample_modality_dropout
from .masking import block_mask, bernoulli_mask, crop, make_mask, random_mask
from .model import DROPPED, MBT, MISSING, MLPA, MLPAOutput, ModalityConfig, PRESENT, normalized_mse

__version__ = '0.1.0'
__all__ = ['MLPA', 'MBT', 'ModalityConfig', 'MLPAOutput', 'PRESENT', 'MISSING', 'DROPPED', 'normalized_mse',
           'sample_modality_dropout', 'make_mask', 'random_mask', 'bernoulli_mask', 'block_mask', 'crop']
