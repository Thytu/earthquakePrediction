import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, intput_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(intput_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(intput_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)

        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)