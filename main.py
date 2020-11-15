# resultat incoherent car mauvais optimizer (SGD à la place d'Adam)
# comment checker l'accuracy ? -> ecart moyen
# issue dans la shape
import rnn
import utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

dataset, labels = utils.load_dataset("./dataset.csv")
dataset, labels = utils.create_seq(dataset, labels, utils.SEQU_SIZE)
X_train, y_train, X_test, y_test = utils.split(dataset, labels, 0.0)

rnn = rnn.RNN(1, 50, 1)

criterion = nn.MSELoss()
learning_rate = 0.001
# optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) # TODO: check wich to take
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
EPOCHS = 2


def train(t, label):
    hidden = rnn.init_hidden()

    for i in range(t.size()[0]):
        # print("T=", t[0][i].unsqueeze(dim=0).unsqueeze(dim=0))
        # print("HIDDEN=", hidden)
        # print(t[0][i].unsqueeze(0).unsqueeze(0))
        output, hidden = rnn(t[0][i].unsqueeze(0).unsqueeze(0), hidden) # TODO: fix

    optimizer.zero_grad()
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    return output, loss

def test(t, label):
    hidden = rnn.init_hidden()
    with torch.no_grad():
        for i in range(t.size()[0]):
            output, hidden = rnn(t[i], hidden)

        loss = criterion(output, label)
        loss.backward()

    return output, loss

# hidden = rnn.init_hidden()
# print(X_train[0].size(), hidden.size())
# train(X_train[0], hidden)

current_loss = 0
curr_diff = 0
all_diff = []
all_losees = []

for i in range(EPOCHS):
    print("epoch:", i+1)
    for index, data in enumerate(X_train):

        output, loss = train(data, y_train[index])
        # exit((curr_diff,  y_train[index].item(), output.item()))

        curr_diff += abs(output.item() - y_train[index].item())
        if (index+1) % 1_000 == 0:
            all_diff.append(curr_diff/1_000)
            curr_diff = 0
        current_loss += loss

        if (index+1) % 1_000 == 0:
            all_losees.append(current_loss/1_000)
            current_loss = 0

plt.figure()
plt.subplots_adjust(hspace=0.35)
plt.subplot(211)
plt.title("Evolution of mean diff over time")
plt.plot(all_diff)
plt.subplot(212)
plt.title("Evolution of mean loss over time")
plt.plot(all_losees)
plt.show()