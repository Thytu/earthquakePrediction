# change softmax to relu
# resultats incoherents car mauvais optimizer (SGD à la place d'Adam)
# comment checker l'accuracy ? -> ecart moyen
# issue dans la shape
import models
import utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

dataset, labels = utils.load_dataset("./dataset.csv")
dataset, labels = utils.create_seq(dataset, labels, utils.SEQU_SIZE)
X_train, y_train, X_test, y_test = utils.split(dataset, labels, 0.5)

rnn = models.RNN(1, 50, 1)

criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
EPOCHS = 2


def train(t, label):
    hidden = rnn.init_hidden()

    for i in range(t.size()[0]):
        output, hidden = rnn(t[0][i].unsqueeze(0).unsqueeze(0), hidden) # TODO: fix dataset shape

    optimizer.zero_grad()
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    return output, loss

def test(t, label):
    hidden = rnn.init_hidden()
    with torch.no_grad():
        for i in range(t.size()[0]):
            output, hidden = rnn(t[0][i].unsqueeze(0).unsqueeze(0), hidden) # TODO: fix dataset shape

        loss = criterion(output, label)
        # loss.backward()

    return output, loss

PLOT_EVERY = 1_000
curr_train_diff = 0
curr_train_loss = 0
curr_test_diff = 0
curr_test_loss = 0

train_diff = []
train_losees = []
test_diff = []
test_losees = []

for i in range(EPOCHS):
    print("epoch:", i+1)
    for index, data in enumerate(X_train):

        output, loss = train(data, y_train[index])

        curr_train_diff += abs(output.item() - y_train[index].item())
        if (index+1) % PLOT_EVERY == 0:
            train_diff.append(curr_train_diff/PLOT_EVERY)
            curr_train_diff = 0
        curr_train_loss += loss

        if (index+1) % PLOT_EVERY == 0:
            train_losees.append(curr_train_loss/PLOT_EVERY)
            curr_train_loss = 0

    for index, data in enumerate(X_test):

        output, loss = test(data, y_test[index])

        curr_test_diff += abs(output.item() - y_test[index].item())
        if (index+1) % PLOT_EVERY == 0:
            test_diff.append(curr_test_diff/PLOT_EVERY)
            curr_test_diff = 0
        curr_test_loss += loss

        if (index+1) % PLOT_EVERY == 0:
            test_losees.append(curr_test_loss/PLOT_EVERY)
            curr_test_loss = 0


plt.figure()
plt.title("Evolution of mean loss over time")
plt.plot(train_losees)
plt.plot(test_losees)
plt.show()

plt.figure()
plt.title("Evolution of mean diff over time")
plt.plot(train_diff)
plt.plot(test_diff)
plt.show()