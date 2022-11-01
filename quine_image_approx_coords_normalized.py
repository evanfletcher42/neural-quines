import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import copy


torch.manual_seed(1337)
random.seed(1337)

target = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(target)


class QuineImageApproxNet(nn.Module):
    def __init__(self):
        super(QuineImageApproxNet, self).__init__()

        self.linear_size = 16

        self.in_proj = nn.Linear(2, self.linear_size, bias=False)
        self.a1 = nn.ReLU(inplace=True)
        self.linear = nn.Linear(self.linear_size, self.linear_size, bias=False)
        self.bn = nn.BatchNorm1d(self.linear_size, affine=False, track_running_stats=False)
        self.a2 = nn.ReLU(inplace=True)
        self.out_proj = nn.Linear(self.linear_size, 1, bias=False)

        # Prevent training input and output projections - we want them to be random and fixed
        for param in self.in_proj.parameters():
            param.requires_grad = False

        for param in self.out_proj.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.in_proj(x)
        x = self.a1(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.a2(x)
        x = self.out_proj(x)
        return x


def train(model: QuineImageApproxNet):
    model.to(device)
    model.train()

    best_model = copy.deepcopy(model)
    best_loss = torch.inf

    weight_lossfunc = nn.MSELoss()

    # Input coordinates, normalized [-1...1].  Map 1:1 with model weights.
    in_r, in_c = torch.meshgrid(torch.linspace(-1, 1, model.linear.weight.size()[0], device=device),
                                torch.linspace(-1, 1, model.linear.weight.size()[1], device=device),
                                indexing='ij')

    in_rc = torch.stack([in_r.flatten(), in_c.flatten()], dim=-1)

    optimizer = optim.RMSprop(model.parameters())

    for epoch in range(20000):

        if epoch % 2 == 0:
            # Regeneration: Set model weights to y_pred
            with torch.no_grad():
                y_pred_w = model(in_rc).reshape(model.linear.weight.size())
                model.linear.weight = torch.nn.Parameter(y_pred_w)

        y_true = model.linear.weight.flatten()[..., None]

        y_pred = model(in_rc)
        loss = weight_lossfunc(y_pred, y_true)

        if loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = loss

        if epoch % 100 == 0:
            print("Epoch ", epoch, " loss ", loss.item())

            if loss.item() == 0.0:
                break

        if epoch % 2 == 1:
            loss.backward()
            optimizer.step()

    model = best_model
    model.eval()

    model_weights = model.linear.weight.detach().cpu().numpy()
    predicted_weights = model(in_rc).reshape(model.linear.weight.shape).detach().cpu().numpy()

    plt_min = min(model_weights.min(), predicted_weights.min())
    plt_max = max(model_weights.max(), predicted_weights.max())

    fig, ax = plt.subplots(1, 3)

    c = ax[0].imshow(model_weights, vmin=plt_min, vmax=plt_max)
    plt.colorbar(c, ax=ax[0])
    ax[0].set_title("Model Weights")

    c = ax[1].imshow(predicted_weights, vmin=plt_min, vmax=plt_max)
    plt.colorbar(c, ax=ax[1])
    ax[1].set_title("Predicted Weights")

    error = model_weights - predicted_weights
    sc = np.max(np.abs(error))
    c = ax[2].imshow(error, cmap='seismic', vmin=-sc, vmax=sc)
    plt.colorbar(c, ax=ax[2])
    rmse = np.sqrt(np.mean(np.square(error)))
    ax[2].set_title(f"RMSE: {rmse:.6e}")

    plt.suptitle("Querying (row, column) w/ Parameter-Free Batch Normalization")

    plt.show()


if __name__ == "__main__":
    model = QuineImageApproxNet()
    train(model)
