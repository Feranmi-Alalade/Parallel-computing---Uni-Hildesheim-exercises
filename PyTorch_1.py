import torch
import numpy as np
import matplotlib.pyplot as plt

X = torch.arange(0.0, 1.0, step=0.01)
X2 = torch.randint(2, size=(len(X),))
Y = (X * 0.7 + X2 * 0.2- 0.3) + torch.normal(0, 0.1, size=(len(X),))

a = torch.tensor([-0.5,], requires_grad=True)
b = torch.tensor([2.0,], requires_grad=True)

opt = torch.optim.SGD([a, b], lr=0.02)
loss = torch.nn.MSELoss()

loss_curve = []

for i in range(50):
    opt.zero_grad()
    y_pred = (X*a) + b
    loss_val = loss(y_pred, Y)
    loss_val.backward()
    opt.step()
    loss_curve.append(loss_val.detach())

plt.plot(loss_curve)
plt.show()