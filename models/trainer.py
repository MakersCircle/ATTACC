# models/trainer.py

from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_loader import VideoDataset
from models.architecture.accident_model import GraphTransformerAccidentModel
from models.loss_tug import TUGLLoss
from models.metrics import compute_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        obj_det, obj_depth, obj_feat, frame_feat, bin_labels, tta, cls = [b.to(device) for b in batch]

        # Forward pass
        prob, uncertainty = model(obj_feat, obj_depth)  # (B, T)

        # Loss
        loss = criterion(prob, uncertainty, bin_labels, tta, cls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Show live batch loss
        progress.set_postfix(batch_loss=f"{loss.item():.4f}")


    return total_loss / len(dataloader)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTransformerAccidentModel().to(device)
    dataset = VideoDataset(is_train=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    criterion = TUGLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    save_dir = Path(config['save_path'])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['epochs']):
        print(f"\n[Epoch {epoch+1}/{config['epochs']}]")
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        if (epoch + 1) % config['save_freq'] == 0:
            save_file = save_dir / f"model_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), save_file)
            print(f"✅ Model saved to {save_file}")

if __name__ == '__main__':
    config = {
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 20,
        'save_freq': 5,
        'batch_size': 4,
        'save_path': './models/saved_models/ccd'
    }

    train_model(config)
