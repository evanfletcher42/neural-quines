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


class QuineOneHotMatrixNet(nn.Module):
    def __init__(self):
        super(QuineOneHotMatrixNet, self).__init__()

        self.linear_size = 64

        # Note:
        self.in_proj = nn.Linear(2*self.linear_size, self.linear_size, bias=True)
        self.a1 = nn.ReLU(inplace=True)
        self.linear = nn.Linear(self.linear_size, self.linear_size, bias=False)
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
        x = self.a2(x)
        x = self.out_proj(x)
        return x


def train(model: QuineOneHotMatrixNet):
    model.to(device)
    model.train()

    lossfunc = nn.MSELoss()

    # Input coordinates.  Map 1:1 with model weights.
    in_r, in_c = torch.meshgrid(torch.arange(model.linear.weight.size()[0], device=device),
                                torch.arange(model.linear.weight.size()[1], device=device),
                                indexing='ij')

    in_rc = torch.stack([in_r.flatten(), in_c.flatten()], dim=-1)
    in_rc = F.one_hot(in_rc)
    in_rc = torch.reshape(in_rc, (-1, model.linear_size*2)).float()

    optimizer = optim.RMSprop(model.parameters())

    # DISABLED: The "best" result per the loss function is visually less compelling than a net allowed to train longer
    # best_model = copy.deepcopy(model)
    # best_loss = torch.inf

    for epoch in range(10000):

        # Regeneration: Set model weights to y_pred
        # DISABLED: Causes convergence to a zero quine
        # with torch.no_grad():
        #     y_pred_r = model(in_rc)
        #     model.linear.weight = torch.nn.Parameter(y_pred_r.reshape(model.linear.weight.size()))

        # Perform a traditional optimization step
        y_true = model.linear.weight.flatten()[..., None]
        y_pred = model(in_rc)

        loss = lossfunc(y_pred, y_true)

        # if loss < best_loss:
        #     best_model = copy.deepcopy(model)
        #     best_loss = loss

        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print("Epoch ", epoch, " loss ", loss.item())

    # model = best_model
    model.eval()

    fig, ax = plt.subplots(1, 3)

    model_weights = model.linear.weight.detach().cpu().numpy()
    predicted_weights = model(in_rc).reshape(model.linear.weight.shape).detach().cpu().numpy()

    plt_min = min(model_weights.min(), predicted_weights.min())
    plt_max = max(model_weights.max(), predicted_weights.max())

    c = ax[0].imshow(model_weights, vmin=plt_min, vmax=plt_max)
    plt.colorbar(c, ax=ax[0])
    ax[0].set_title("Model Weights")

    c = ax[1].imshow(predicted_weights, vmin=plt_min, vmax=plt_max)
    plt.colorbar(c, ax=ax[1])
    ax[1].set_title("Predicted Weights")

    error = model(in_rc).reshape(model.linear.weight.shape).detach().cpu().numpy() - model.linear.weight.detach().cpu().numpy()
    sc = np.max(np.abs(error))
    c = ax[2].imshow(error, cmap='seismic', vmin=-sc, vmax=sc)
    plt.colorbar(c, ax=ax[2])
    ax[2].set_title("Error (RMSE %f)" % np.sqrt(np.mean(np.square(error))))

    plt.suptitle("One-Hot Model, RMSprop")
    plt.show()


if __name__ == "__main__":
    model = QuineOneHotMatrixNet()
    train(model)
