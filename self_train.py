import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import AlphaGoZeroNet


class GoDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, pi, v = self.data[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor([v], dtype=torch.float32)
        )


def train_model(data_path="self_play_data.pt",
                base_model_path="alphago_model_v3_candidate.pth",
                save_model_path="hybrid.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hybrid Training starting... Hardware: {device}")

    model = AlphaGoZeroNet(num_res_blocks=20, hidden_channels=256).to(device)
    try:
        # Using the previously trained v3_candidate file as a base
        model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
        print(f"Base intelligence ({base_model_path}) loaded for fine-tuning.")
    except Exception as e:
        print("Warning: Base weights not found. Starting from scratch.")

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    dataset = GoDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 2

    for epoch in range(epochs):
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0

        for states, pis, values in dataloader:
            states, pis, values = states.to(device), pis.to(device), values.to(device)

            optimizer.zero_grad()

            policy_logits, value_pred = model(states)

            # Policy Loss (Cross-Entropy / Log Softmax over 362 probabilities)
            policy_loss = -torch.sum(pis * torch.log_softmax(policy_logits, dim=1)) / states.size(0)

            # Value Loss
            value_loss = mse_loss(value_pred, values)

            # CRITICAL FIX: To ensure the model doesn't forget its old vision during fine-tuning
            # and to balance the weights, we suppress the value weight.
            value_weight = 0.01
            loss = policy_loss + (value_weight * value_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Total Loss: {total_loss / len(dataloader):.4f} "
              f"(Policy: {policy_loss_total / len(dataloader):.4f}, Value: {value_loss_total / len(dataloader):.4f})")

    torch.save(model.state_dict(), save_model_path)
    print(f"\nTraining complete! Brand new V3 model saved as '{save_model_path}'.")


if __name__ == "__main__":
    train_model()