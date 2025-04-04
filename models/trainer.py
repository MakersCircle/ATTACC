# models/trainer.py

from pathlib import Path
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_loader import VideoDataset
from models.architecture.accident_model import GraphTransformerAccidentModel
from models.loss_tug import TUGLLoss
from models.metrics import compute_metrics

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "trainer.log"
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, save_dir):
    model.train()
    total_loss = 0
    batch_losses = []

    progress = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False)
    for batch_idx, batch in enumerate(progress):
        obj_det, obj_depth, obj_feat, frame_feat, bin_labels, tta, cls = [b.to(device) for b in batch]

        try:
            prob, uncertainty = model(obj_feat, obj_depth)

            loss = criterion(prob, uncertainty, bin_labels, tta, cls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            total_loss += loss.item()
            progress.set_postfix(batch_loss=f"{loss.item():.4f}")

        except Exception as e:
            logger.error(f"Error during training at batch {batch_idx}: {str(e)}")
            continue

    loss_save_path = save_dir / f"batch_losses_epoch{epoch + 1}.npy"
    np.save(loss_save_path, np.array(batch_losses))
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    return avg_loss


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        model = GraphTransformerAccidentModel().to(device)
        dataset = VideoDataset(is_train=True)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        criterion = TUGLLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        save_dir = Path(config['save_path'])
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training started for {config['epochs']} epochs")

        epoch_losses = []
        for epoch in range(config['epochs']):
            print(f"\n[Epoch {epoch+1}/{config['epochs']}]")
            logger.info(f"Starting Epoch {epoch+1}")
            train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, save_dir)
            print(f"Train Loss: {train_loss:.4f}")
            epoch_losses.append(train_loss)

            if (epoch + 1) % config['save_freq'] == 0:
                save_file = save_dir / f"model_epoch{epoch+1}.pt"
                torch.save(model.state_dict(), save_file)
                logger.info(f"Model saved at epoch {epoch+1} to {save_file}")
                print(f"✅ Model saved to {save_file}")

        epoch_path = save_dir / "epoch_losses.npy"
        np.save(epoch_path, np.array(epoch_losses))
        logger.info(f"Epoch losses saved to {epoch_path}")

        logger.info("Training complete.")

    except Exception as e:
        logger.error(f"Training failed due to error: {str(e)}")
        print(f"❌ Training failed: {e}")

if __name__ == '__main__':
    config = {
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 40,
        'save_freq': 5,
        'batch_size': 4,
        'save_path': './saved_models/ccd'
    }

    train_model(config)
