import torch
from torch import nn, optim
import torch.utils.data
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import copy

torch.manual_seed(1337)
random.seed(1337)

target = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(target)


class QuineSourceRNN(nn.Module):
    def __init__(self):
        super(QuineSourceRNN, self).__init__()

        self.n_layers = 2
        self.lstm_size = 16

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.lstm_size,
            num_layers=self.n_layers,
        )
        self.fc = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.norm = nn.InstanceNorm1d(self.lstm_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.lstm_size, 1, bias=True)

        # Fixed projection in output layer
        # for param in self.fc.parameters():
        #     param.requires_grad = False

    def forward(self, x, prev_state):
        y, state = self.lstm(x, prev_state)
        y = self.fc(y)
        y = self.norm(y)
        y = self.relu(y)
        y = self.fc2(y)

        y = self.fc(out)

        return y, state

    def init_state(self):
        return (torch.zeros(self.n_layers, self.lstm_size, device=device),
                torch.zeros(self.n_layers, self.lstm_size, device=device))


def train(model):
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    # optimizer = optim.RMSprop(model.parameters())

    best_model = copy.deepcopy(model)
    best_loss = torch.inf

    n_epochs = 2500
    for epoch in range(n_epochs):
        state_h, state_c = model.init_state()
        optimizer.zero_grad()

        # Extract input/output sequence, with initial input of 0
        y = torch.cat([w.flatten() for w in model.parameters()])[..., None]
        x = y.roll(1)

        # Add some random noise to x.
        # Helps the model learn not to rely on precisely-correct inputs.
        # x = x + torch.randn_like(x)*0.001
        x[0, 0] = 0

        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        loss = criterion(y_pred, y)

        if loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = loss

        if epoch % 20 == 0:
            print(f"e {epoch} l {loss:.8e}")

        if epoch == n_epochs-1:
            print("\nWeights (True, Predicted, Error):")
            for i in range(y.shape[0]):
                err = y[i, 0].item() - y_pred[i, 0].item()
                print(y[i, 0].item(), y_pred[i, 0].item(), err)

        loss.backward()
        optimizer.step()
        epoch += 1

    model = best_model


def evaluate(model):
    # ground truth parameters
    gt_flat = torch.cat([w.flatten() for w in model.parameters()])[..., None]
    n_params = gt_flat.shape[0]

    # Extract sequence from the model
    model.eval()

    state_h, state_c = model.init_state()
    x_in = torch.tensor([[0]], dtype=torch.float32, device=device)

    sequence_pred = []
    for i in range(n_params):
        x_in, (state_h, state_c) = model(x_in, (state_h, state_c))
        sequence_pred.append(x_in.item())

    sequence_pred = np.array(sequence_pred)
    gt_flat = gt_flat.detach().cpu().numpy()

    plt_min = min(sequence_pred.min(), gt_flat.min())
    plt_max = max(sequence_pred.max(), gt_flat.max())

    def sqf(x):
        a = int(np.sqrt(x))
        while a > 1:
            if x % a == 0:
                break
            a -= 1

        print(x//a, a)
        return x // a, a

    img_shape = sqf(sequence_pred.shape[0])
    sequence_pred = sequence_pred.reshape(img_shape)

    fig, ax = plt.subplots(1,3)

    gt_flat = gt_flat.reshape(img_shape)
    c = ax[0].imshow(gt_flat, vmin=plt_min, vmax=plt_max)
    plt.colorbar(c, ax=ax[0])
    ax[0].set_title("Model Weights")

    c = ax[1].imshow(sequence_pred, vmin=plt_min, vmax=plt_max)
    plt.colorbar(c, ax=ax[1])
    ax[1].set_title("Predicted Weights")

    error = np.abs(sequence_pred - gt_flat)
    sc = np.max(np.abs(error))
    c = ax[2].imshow(error, cmap='seismic', vmin=-sc, vmax=sc)
    plt.colorbar(c, ax=ax[2])
    rmse = np.sqrt(np.mean(np.square(error)))
    ax[2].set_title(f"RMSE: {rmse:.6e}")

    plt.show()


if __name__ == "__main__":
    model = QuineSourceRNN()
    train(model)
    evaluate(model)
