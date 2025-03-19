import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.architecture.accident_model import AccidentPredictor


class AccidentDataset(Dataset):
    """
    Custom Dataset for Accident Anticipation.
    """

    def __init__(self, f_obj, f_frame, f_depth, labels):
        """
        Args:
            f_obj (torch.Tensor): Object features (batch_size, T, N, D).
            f_frame (torch.Tensor): Frame features (batch_size, T, 1, D).
            f_depth (torch.Tensor): Depth features (batch_size, T, N, d).
            labels (torch.Tensor): Ground truth accident labels (batch_size, T).
        """
        self.f_obj = f_obj
        self.f_frame = f_frame
        self.f_depth = f_depth
        self.labels = labels

    def __len__(self):
        return self.f_obj.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return (
            self.f_obj[idx],
            self.f_frame[idx],
            self.f_depth[idx],
            self.labels[idx]
        )


class AccidentTrainer:
    """
    Trainer for the AccidentPredictor model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, lr=1e-4,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Loss Function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train(self, num_epochs=10, save_path="accident_predictor.pth"):
        """
        Train the model.
        """
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for f_obj, f_frame, f_depth, labels in self.train_loader:
                f_obj, f_frame, f_depth, labels = (
                    f_obj.to(self.device),
                    f_frame.to(self.device),
                    f_depth.to(self.device),
                    labels.to(self.device),
                )

                # Forward pass
                outputs = self.model(f_obj, f_frame, f_depth)

                # Compute loss
                loss = self.criterion(outputs, labels)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            val_loss = self.evaluate()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def evaluate(self):
        """
        Evaluate model on validation set.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for f_obj, f_frame, f_depth, labels in self.val_loader:
                f_obj, f_frame, f_depth, labels = (
                    f_obj.to(self.device),
                    f_frame.to(self.device),
                    f_depth.to(self.device),
                    labels.to(self.device),
                )

                # Forward pass
                outputs = self.model(f_obj, f_frame, f_depth)

                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def test(self, model_path="accident_predictor.pth"):
        """
        Test the trained model on the test set.
        """
        # Load trained model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for f_obj, f_frame, f_depth, labels in self.test_loader:
                f_obj, f_frame, f_depth, labels = (
                    f_obj.to(self.device),
                    f_frame.to(self.device),
                    f_depth.to(self.device),
                    labels.to(self.device),
                )

                # Forward pass
                outputs = self.model(f_obj, f_frame, f_depth)

                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Convert predictions to binary (accident = 1, no accident = 0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.numel()

        accuracy = correct / total * 100
        avg_test_loss = total_loss / len(self.test_loader)

        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        return avg_test_loss, accuracy


# Example Usage
if __name__ == "__main__":
    # Example data dimensions
    batch_size, T, N, D, d = 2, 50, 19, 4096, 16

    # Generate random synthetic data
    f_obj = torch.rand(batch_size * 10, T, N, D)
    f_frame = torch.rand(batch_size * 10, T, 1, D)
    f_depth = torch.rand(batch_size * 10, T, N, d)
    labels = torch.randint(0, 2, (batch_size * 10, T)).float()  # Binary labels (0 or 1)

    # Split into training, validation, and test sets (80-10-10 split)
    train_size = int(0.8 * len(f_obj))
    val_size = int(0.1 * len(f_obj))
    test_size = len(f_obj) - train_size - val_size

    train_dataset = AccidentDataset(f_obj[:train_size], f_frame[:train_size], f_depth[:train_size], labels[:train_size])
    val_dataset = AccidentDataset(f_obj[train_size:train_size + val_size], f_frame[train_size:train_size + val_size],
                                  f_depth[train_size:train_size + val_size], labels[train_size:train_size + val_size])
    test_dataset = AccidentDataset(f_obj[train_size + val_size:], f_frame[train_size + val_size:],
                                   f_depth[train_size + val_size:], labels[train_size + val_size:])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and trainer
    model = AccidentPredictor(D, d)
    trainer = AccidentTrainer(model, train_loader, val_loader, test_loader)

    # Train model
    trainer.train(num_epochs=10)

    # Test model
    trainer.test()
