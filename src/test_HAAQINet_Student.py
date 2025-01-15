import torch
import torchaudio
import argparse
import numpy as np
from pathlib import Path
import yaml
import json
from typing import List, Tuple
import logging
import torchaudio.compliance.kaldi as ta_kaldi
import teacher_module_ws as teacher_module
from BEATs import BEATs, BEATsConfig


def setup_logger() -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger('HAAQI-Net-Test')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def extract_features(audio_path: str) -> torch.Tensor:
    """
    Extract log-mel spectrogram features from audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tensor of shape (1, freq_bins, time_frames)
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Extract log-mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64
    )(waveform)
    
    # Convert to log scale
    log_mel = torch.log1p(mel_spectrogram)
    
    # Add batch dimension
    log_mel = log_mel.unsqueeze(0)
    
    return log_mel


def preprocess(
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        ) -> torch.Tensor:
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank


def load_BEATs(beats_model_path):
    # load the pre-trained checkpoints
    checkpoint = torch.load(beats_model_path)

    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    
    # Detach gradients before returning the model
    for param in BEATs_model.parameters():
        param.requires_grad_(False)
    
    return BEATs_model


def load_model(model_path: str, config_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """
    Load HAAQI-Net model and configuration.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to model configuration file
        device: torch device
        
    Returns:
        Tuple of (model, config)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    from HAAQI_Net import HAAQINetStudent  # Import your model class
    model = HAAQINetStudent(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        linear_output=config['model']['linear_output'],
        act_fn=config['model']['act_fn']
    )

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, config


def predict(
    audio_path: str,
    hearing_levels: List[float],
    model_path: str,
    config_path: str,
    device: torch.device
) -> Tuple[float, np.ndarray]:
    """
    Generate HAAQI predictions for an audio file.

    Args:
        audio_path: Path to audio file
        hearing_levels: List of hearing levels for 8 frequency bands
        model_path: Path to model checkpoint
        config_path: Path to model configuration
        device: torch device

    Returns:
        Tuple of (average_score, frame_scores)
    """
    # Load model
    model, config = load_model(model_path, config_path, device)

    # Extract features
    waveform, org_sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=org_sr, new_freq=16000)
    
    fbank = preprocess(source=waveform)
    fbank = fbank.to(device)

    # Prepare hearing levels
    hl = torch.tensor(hearing_levels, dtype=torch.float32)
    hl = hl.unsqueeze(0).to(device)  # Add batch dimension

    # Generate predictions
    with torch.no_grad():
        frame_scores, avg_score, *_ = model(fbank.unsqueeze(0), hl)

    return avg_score.item(), frame_scores.squeeze().cpu().numpy()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test HAAQI-Net on audio file')
    parser.add_argument('--audio', type=str, required=False,
                      help='Path to audio file')
    parser.add_argument('--hl', type=str, required=False,
                      help='Hearing levels as JSON array [freq1, freq2, ..., freq8]')
    parser.add_argument('--model', type=str, default='model/haaqi_net_distillBEATs.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='src/config.yaml',
                      help='Path to model configuration file')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Parse hearing levels
    try:
        hearing_levels = json.loads(args.hl)
        assert len(hearing_levels) == 8, "Must provide 8 hearing levels"
    except (json.JSONDecodeError, AssertionError) as e:
        logger.error(f"Error parsing hearing levels: {e}")
        logger.error("Please provide hearing levels as a JSON array with 8 values")
        return

    # Check if audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    # Generate predictions
    try:
        avg_score, frame_scores = predict(
            str(audio_path),
            hearing_levels,
            args.model,
            args.config,
            device
        )

        # Print results
        logger.info(f"Audio file: {audio_path.name}")
        logger.info(f"Hearing levels: {hearing_levels}")
        logger.info(f"Average HAAQI score: {avg_score:.3f}")
        logger.info(f"Frame scores shape: {frame_scores.shape}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")


if __name__ == '__main__':
    main()