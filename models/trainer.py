# models/trainer.py

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

    for batch in dataloader:
        obj_det, obj_depth, obj_feat, frame_feat, bin_labels, tta, cls = [b.to(device) for b in batch]

        prob, uncertainty = model(obj_feat, obj_depth)
        loss = criterion(prob, uncertainty, bin_labels, tta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTransformerAccidentModel().to(device)
    dataset = VideoDataset(is_train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    criterion = TUGLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    for epoch in tqdm(range(config['epochs'])):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

        if (epoch + 1) % config['save_freq'] == 0:
            torch.save(model.state_dict(), f"{config['save_path']}/model_epoch{epoch+1}.pt")


if __name__ == '__main__':
    config = {
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 20,
        'save_freq': 5,
        'save_path': './models/saved_models/ccd'
    }

    train_model(config)