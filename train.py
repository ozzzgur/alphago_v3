import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_processor import parse_sgf_folder
from model import AlphaGoZeroNet


def train_model(data_folder, epochs=10, batch_size=128, max_games=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V3 Model is being trained on {device} with PURE FP32 (32-bit) precision...")

    print("\nPreparing KGS Dataset in V3 standards (17 Channels)...")
    X, y_policy, y_value = parse_sgf_folder(data_folder, max_games=max_games)

    print(f"\n=== TRAINING DATA SUMMARY ===")
    print(f"Total Move (Snapshot) Count: {X.shape[0]}")
    print(f"Tensor Shape: {X.shape}")

    dataset = TensorDataset(X, y_policy, y_value)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Calling the new model with 17 channels and 362 outputs
    model = AlphaGoZeroNet(num_res_blocks=20, hidden_channels=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    print("\nTraining loop starting...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0

        for batch_X, batch_policy, batch_value in dataloader:
            batch_X = batch_X.to(device)
            batch_policy = batch_policy.to(device)
            batch_value = batch_value.to(device).unsqueeze(1)

            optimizer.zero_grad()

            pred_policy, pred_value = model(batch_X)

            loss_policy = policy_criterion(pred_policy, batch_policy)
            loss_value = value_criterion(pred_value, batch_value)

            # CRITICAL FIX: To prevent the Value network from blindly memorizing
            # when training with human data, we set its weight to 0.01.
            value_weight = 0.01
            loss = loss_policy + (value_weight * loss_value)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            policy_loss_total += loss_policy.item()
            value_loss_total += loss_value.item()

        avg_loss = total_loss / len(dataloader)
        avg_pol = policy_loss_total / len(dataloader)
        avg_val = value_loss_total / len(dataloader)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | Average Total Loss: {avg_loss:.4f} (Policy: {avg_pol:.4f}, Value: {avg_val:.4f})")

    # Saving the V3 model
    model_path = "alphago_model_v3_candidate.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete! Brand new V3 model weights saved: {model_path}")


if __name__ == "__main__":
    # Quick foundation building with KGS data
    train_model("kgs_data", epochs=5, batch_size=128, max_games=5000)