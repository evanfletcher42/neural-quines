import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
import torch.nn.functional as F
import random

torch.manual_seed(1337)
random.seed(1337)

target = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(target)

START = 128
END = 129


class QuineSourceRNN(nn.Module):
    def __init__(self):
        super(QuineSourceRNN, self).__init__()

        self.n_layers = 2
        self.n_vocab = 130
        self.lstm_size = 128
        self.embed = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=32)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=self.lstm_size,
            num_layers=self.n_layers,
        )
        self.fc = nn.Linear(self.lstm_size, self.n_vocab)

    def forward(self, x, prev_state):
        e = self.embed(x)
        out, state = self.lstm(e, prev_state)
        logits = self.fc(out)

        return logits, state

    def init_state(self):
        return (torch.zeros(self.n_layers, self.lstm_size, device=device),
                torch.zeros(self.n_layers, self.lstm_size, device=device))


def encode_str(y_str):
    return torch.tensor([START] + [ord(x) for x in y_str] + [END], device=device)


def load_data():
    with open(__file__, "r") as f:
        y_str = f.read()
        return encode_str(y_str)


def train(model):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

    y_enc = load_data()
    x = y_enc[:-1]
    y = y_enc[1:]

    epoch = 0
    while True:
        state_h, state_c = model.init_state()
        optimizer.zero_grad()

        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        loss = criterion(y_pred, y)
        acc = torch.sum(torch.eq(torch.argmax(y_pred, dim=-1), y))

        if epoch % 20 == 0:
            print("# e %d l %f a %d/%d" % (epoch, loss, acc, y.shape[0]))

        if acc == y.shape[0]:
            print("# Done!\n")
            break

        loss.backward()
        optimizer.step()
        epoch += 1


def predict(model):
    model.eval()
    state_h, state_c = model.init_state()
    in_char = torch.tensor([START], device=device)
    while True:
        y_pred, (state_h, state_c) = model(in_char, (state_h, state_c))
        yp = F.softmax(y_pred, dim=1)
        yi = torch.argmax(yp)
        yi_c = yi.cpu().item()
        if yi_c == END:
            break
        print(chr(yi_c), end='')
        in_char[0] = yi
    print()


if __name__ == "__main__":
    model = QuineSourceRNN()
    model.to(device)
    train(model)
    predict(model)
