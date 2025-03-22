# models/evaluator.py

import torch
from torch.utils.data import DataLoader
from data.data_dataloader import VideoDataset
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

    for batch in dataloader:
        obj_det, obj_depth, obj_feat, frame_feat, bin_labels, tta, cls = [b.to(device) for b in batch]

        with torch.no_grad():
            probs, _ = model(obj_feat, obj_depth)
            probs = probs.detach().cpu().numpy()
            labels = bin_labels.squeeze().cpu().numpy()

        metrics = compute_metrics(probs, labels)
        results.append(metrics)

    # Aggregate metrics
    ap_scores = [r['AP'] for r in results]
    tta_scores = [r['mTTA'] for r in results if r['mTTA'] > 0]

    print(f"Mean AP  : {np.mean(ap_scores):.4f}")
    print(f"Mean TTA : {np.mean(tta_scores):.2f} frames")
