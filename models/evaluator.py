# models/evaluator.py

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset_loader import VideoDataset
from models.architecture.accident_model import GraphTransformerAccidentModel
from models.metrics import compute_metrics

def evaluate_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTransformerAccidentModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = VideoDataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    output = {}

    for batch in dataloader:

        obj_det, obj_depth, obj_feat, frame_feat, bin_labels, tta, cls, vid_name = batch
        obj_det, obj_depth, obj_feat, frame_feat, bin_labels, tta, cls = [b.to(device) for b in [obj_det, obj_depth, obj_feat, frame_feat, bin_labels, tta, cls]]

        with torch.no_grad():
            probs, _ = model(obj_feat, obj_depth)
            probs = probs.detach().cpu().numpy()
            labels = bin_labels.squeeze().cpu().numpy()


        for i in range(len(probs)):
            output[f'{int(cls[i].item())}_{vid_name[i]}'] = {
                'probabilities': probs[i],
                'labels': bin_labels[i].cpu().numpy(),
                'cls': cls[i].cpu().numpy(),
            }

        metrics = compute_metrics(probs, labels)
        results.append(metrics)

    output_path = Path(__file__).parent.parent / 'outputs' / 'testing' / 'ccd' / 'predictions.npz'
    np.savez(output_path, **output)

    # Aggregate metrics
    ap_scores = [r['AP'] for r in results]
    tta_scores = [r['mTTA'] for r in results if r['mTTA'] > 0]

    print(f"Mean AP  : {np.mean(ap_scores):.4f}")
    print(f"Mean TTA : {np.mean(tta_scores):.2f} frames")


if __name__ == "__main__":
    model_path = Path(__file__).parent / 'saved_models' / 'ccd' / 'GATConv_TUGL' / 'model_epoch40.pt'
    evaluate_model(model_path=model_path)