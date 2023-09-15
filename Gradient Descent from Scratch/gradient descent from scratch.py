import torch
import numpy as np


# model output
def forward(X, w):
    return w*X

# Mean Squared Error
def loss(y_pred, y_hat):
    # calculate MSE
    return ((y_pred-y_hat)**2).mean()

# gradient of the loss with respect to w
def gradient(X, y_pred, y_hat):
    return np.mean(2*X*(y_pred-y_hat))


# the model is a linear regression: y = w.x^T = w_1*x_1 + w_2*x_2
w_actual = 2

# input data
X = np.random.rand(10,)
Y = X * w_actual


### training
# intialize w
w = 0

n_epoch = 10
learning_rate = 0.01

for epoch in range(n_epoch):
    y_pred = forward(X, w)
    l = loss(y_pred, Y)
    w -= learning_rate*gradient(X, y_pred, Y)
    if epoch % 100:
        print(f'epoch {epoch}, loss {l:.10f}')


######################## using pytorch #####################################3

import torch
import torch.nn as nn


X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
n_sample, n_feature = X.shape
model = nn.Linear(n_feature, n_feature)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

n_epoch = 1000

for epoch in range(n_epoch):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 20 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch}, loss {l:.10f}, parameters: {w[0][0]}')




