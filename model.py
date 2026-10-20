import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """A standard Residual Block to extract deep features from the board."""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaGoZeroNet(nn.Module):
    """Dual-Head ResNet architecture for Go (V3: 17 Channels, 362 Actions)."""

    # V3 Update: input_channels increased from 3 to 17.
    def __init__(self, input_channels=17, board_size=19, num_res_blocks=10, hidden_channels=128):
        super(AlphaGoZeroNet, self).__init__()
        self.board_size = board_size

        # Action Space: 19x19 (361) Board Moves + 1 Pass Move = 362
        self.action_space = (board_size * board_size) + 1

        # 1. Initial Convolutional Block
        self.conv_input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(hidden_channels)

        # 2. Residual Tower (Backbone)
        self.res_tower = nn.ModuleList([ResBlock(hidden_channels) for _ in range(num_res_blocks)])

        # 3. Policy Head (Predicts move probabilities)
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        # V3 Update: Output size set to board_size * board_size + 1 (362).
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_space)

        # 4. Value Head (Predicts win rate: -1 to 1)
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Pass through the initial block
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Pass through the residual tower
        for block in self.res_tower:
            x = block(x)

        # --- Policy Head ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(p)

        # --- Value Head ---
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        value_out = torch.tanh(self.value_fc2(v))

        return policy_logits, value_out


# For testing purposes:
if __name__ == "__main__":
    # V3 Update: Test tensor is now 17-channel
    dummy_input = torch.randn(6, 17, 19, 19)
    model = AlphaGoZeroNet()
    policy, value = model(dummy_input)

    print("=== MODEL OUTPUT SHAPES ===")
    print(f"Policy Output : {policy.shape} -> (Batch, 362 moves)")
    print(f"Value Output  : {value.shape} -> (Batch, 1 outcome)")