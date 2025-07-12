############################################################## Regression problem

# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import pandas as pd
# import torch

# # Load the wine dataset
# wine = load_wine()

# # Convert to pandas DataFrame for easier viewing
# X = wine.data
# y = wine.target.astype(float)

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
# y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

# model = torch.nn.Sequential(
#     torch.nn.Linear(13,32),
#     torch.nn.ReLU(),
#     torch.nn.Linear(32,16),
#     torch.nn.ReLU(),
#     torch.nn.Linear(16,1)
# )

# loss = torch.nn.MSELoss()
# opt = torch.optim.SGD(model.parameters(), lr=0.01)

# loss_values = []

# for epoch in range(50):
#     model.train()
#     opt.zero_grad()
#     output = model(X_train)
#     train_loss = loss(output, y_train)
#     train_loss.backward()
#     opt.step()
#     loss_values.append(train_loss.detach())

# plt.plot(loss_values)
# plt.xlabel("Epochs")
# plt.ylabel("MSE loss")
# plt.grid()
# plt.show()

# model.eval()
# with torch.no_grad():
#     y_pred = model(X_test)
#     test_loss = loss(y_pred, y_test)

# print(f"Test loss is {test_loss.detach()}")


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Load the wine dataset
wine = load_wine()

# Convert to pandas DataFrame for easier viewing
X = wine.data
y = wine.target.astype(float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

model = torch.nn.Sequential(
    torch.nn.Linear(13,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,3)
)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

loss_values = []

for epoch in range(50):
    model.train()
    opt.zero_grad()
    output = model(X_train)
    train_loss = loss(output, y_train)
    train_loss.backward()
    opt.step()
    loss_values.append(train_loss.detach())

plt.plot(loss_values)
plt.xlabel("Epochs")
plt.ylabel("CrossEntropy loss")
plt.grid()
plt.show()

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = loss(y_pred, y_test)

    predicted_classes = torch.argmax(y_pred, dim=1)
    accuracy = (predicted_classes == y_test).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")

    print(f"Test loss is {test_loss.detach()}")









