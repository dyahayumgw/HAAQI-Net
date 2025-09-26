import os
import torch
import torchaudio
import argparse
import yaml
import json
import logging
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Tuple
from BEATs import BEATs, BEATsConfig
from HAAQI_Net import HAAQINet  # Import your model class


def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f2:
                l = yaml.safe_load(f2)
                cfg.update(l)
    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg


class AudioDataset(Dataset):
    """Custom Dataset for loading audio files and corresponding labels."""
    def __init__(self, file_paths: List[str], audiograms: List[List[float]], haaqi_scores: List[float]):
        self.file_paths = file_paths
        self.audiograms = audiograms
        self.haaqi_scores = haaqi_scores

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        audiogram = self.audiograms[idx]
        haaqi_score = self.haaqi_scores[idx]

        # Load and preprocess the audio file
        waveform, sr = torchaudio.load(audio_path)           # [C, T]
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        return waveform, torch.tensor(audiogram, dtype=torch.float32), torch.tensor(haaqi_score, dtype=torch.float32)


def setup_logger() -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger('HAAQI-Net-Train')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_BEATs(beats_model_path):
    # load the pre-trained checkpoints
    checkpoint = torch.load(beats_model_path, map_location="cpu")
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
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    beats_model = load_BEATs(config['beats_model_path'])
    beats_model.to(device)

    model = HAAQINet(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        linear_output=config['model']['linear_output'],
        act_fn=config['model']['act_fn'],
        beats_model=beats_model
    )

    # Load model weights (if provided)
    if model_path and Path(model_path).is_file():
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    model.to(device)
    model.train()

    return model, config


class EarlyStopping:
    """Stop when validation loss doesn't improve for `patience` epochs."""
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@torch.no_grad()
def validate(model, dataloader, device, loss_fn) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for waveform, audiogram, haaqi_score in dataloader:
        waveform, audiogram, haaqi_score = waveform.to(device), audiogram.to(device), haaqi_score.to(device)

        # Keep it simple: loop over items so the model can be called the same way as before
        batch_losses = []
        for b in range(waveform.size(0)):
            w = waveform[b]              # [C, T]
            a = audiogram                # [1, 8]
            y = haaqi_score              # [1]
            haaqi_frame, haaqi_avg = model(w, a)
            batch_losses.append(loss_fn(haaqi_avg, y) + loss_fn(haaqi_frame, y))
        loss = torch.stack(batch_losses).mean()

        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, patience=3, save_path=None, logger=None):
    if logger is None:
        logger = setup_logger()
    early = EarlyStopping(patience=patience)
    best_val = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for waveform, audiogram, haaqi_score in train_loader:
            waveform, audiogram, haaqi_score = waveform.to(device), audiogram.to(device), haaqi_score.to(device)

            optimizer.zero_grad()

            # Keep behavior close to original but batch-safe
            batch_losses = []
            for b in range(waveform.size(0)):
                w = waveform[b]          # [C, T]
                a = audiogram            # [1, 8]
                y = haaqi_score          # [1]
                haaqi_frame, haaqi_avg = model(w, a)
                batch_losses.append(loss_fn(haaqi_avg, y) + loss_fn(haaqi_frame, y))

            loss = torch.stack(batch_losses).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_train = total_loss / max(steps, 1)

        if val_loader is not None:
            val_loss = validate(model, val_loader, device, loss_fn)
            logger.info(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_train:.4f}  Val Loss: {val_loss:.4f}")

            # Save best
            if val_loss < best_val and save_path is not None:
                best_val = val_loss
                torch.save({"model": model.state_dict()}, save_path)
                logger.info(f"Saved best model to {save_path} (val_loss={val_loss:.4f})")

            # Early stopping
            if early.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience}).")
                break
        else:
            logger.info(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_train:.4f}")

    # If we never validated / didn't save best during training, save final
    if val_loader is None and save_path is not None:
        torch.save({"model": model.state_dict()}, save_path)
        logger.info(f"Training complete, model saved to {save_path}.")


def main(config_path):
    parser = argparse.ArgumentParser(description='Train HAAQI-Net')
    config = yaml_config_hook(config_path)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load train JSON
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)["data"]
    file_paths = [item["file_path"] for item in train_data]
    audiograms = [item["audiogram"] for item in train_data]
    haaqi_scores = [item["haaqi_score"] for item in train_data]

    train_ds = AudioDataset(file_paths, audiograms, haaqi_scores)

    # Optional validation JSON (only if present in config)
    val_loader = None
    if hasattr(args, "val_data") and args.val_data:
        with open(args.val_data, 'r') as f:
            val_data = json.load(f)["data"]
        v_fp = [item["file_path"] for item in val_data]
        v_ag = [item["audiogram"] for item in val_data]
        v_hq = [item["haaqi_score"] for item in val_data]
        val_ds = AudioDataset(v_fp, v_ag, v_hq)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=False,
                                num_workers=args.num_workers, shuffle=False)

    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=True,
                              num_workers=args.num_workers, shuffle=True)

    model, _ = load_model(args.model_path, config_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Patience fixed to 3 as requested
    train(model, train_loader, val_loader, optimizer, loss_fn, device,
          epochs=args.epochs, patience=3, save_path=args.save_model_path, logger=logger)

    # If validation existed, best model was already saved during training.
    if val_loader is not None:
        logger.info("Training complete (best model saved during validation).")
    else:
        logger.info("Training complete and model saved.")


if __name__ == '__main__':
    config_path = "/src/config.yaml"
    print(config_path)
    main(config_path)
