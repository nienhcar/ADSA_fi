import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, f1_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt

class My_Reg(nn.Module):
    def __init__(self, input_dim):
        super(My_Reg, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single neuron for regression
        )

    def forward(self, x):
        return self.model(x)

class My_Clf(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(My_Clf, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # Number of classes
        )

    def forward(self, x):
        return self.model(x)

def train_regression_model(X, y, learning_rate=0.01, epochs=200):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    model = My_Reg(X.shape[1])
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        optimizer = optimizer_sgd if epoch % 2 == 0 else optimizer_adagrad
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        l1_lambda, l2_lambda = 0.01, 0.01
        l1_reg, l2_reg = torch.tensor(0.), torch.tensor(0.)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)
        loss += l1_lambda * l1_reg + l2_lambda * l2_reg

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

def train_classification_model(X, y, num_classes=2, learning_rate=0.01, epochs=200):
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    model = My_Clf(X.shape[1], num_classes)
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        optimizer = optimizer_sgd if epoch % 2 == 0 else optimizer_adagrad
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        l1_lambda, l2_lambda = 0.01, 0.01
        l1_reg, l2_reg = torch.tensor(0.), torch.tensor(0.)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)
        loss += l1_lambda * l1_reg + l2_lambda * l2_reg

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

def evaluate_regression(model, X, y):
    X = torch.FloatTensor(X)
    predictions = model(X).detach().numpy()
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return {'R2': r2, 'RMSE': rmse, 'Adjusted R2': adjusted_r2}

def evaluate_classification(model, X, y, num_classes=2):
    X = torch.FloatTensor(X)
    outputs = model(X)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.numpy()
    if num_classes == 2:
        return {'F1 Score': f1_score(y, predicted), 'Confusion Matrix': confusion_matrix(y, predicted)}
    else:
        return {
            'Micro F1': f1_score(y, predicted, average='micro'),
            'Macro F1': f1_score(y, predicted, average='macro'),
            'MCC': matthews_corrcoef(y, predicted)
        }
california = fetch_california_housing()
cancer = load_breast_cancer()
X_multi, y_multi = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=4, random_state=42)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(california.data, california.target, test_size=0.2)
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(cancer.data, cancer.target, test_size=0.2)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.2)

scaler = StandardScaler()
X_reg_train, X_reg_test = scaler.fit_transform(X_reg_train), scaler.transform(X_reg_test)
X_class_train, X_class_test = scaler.fit_transform(X_class_train), scaler.transform(X_class_test)
X_multi_train, X_multi_test = scaler.fit_transform(X_multi_train), scaler.transform(X_multi_test)

reg_model, reg_losses = train_regression_model(X_reg_train, y_reg_train)
class_model, class_losses = train_classification_model(X_class_train, y_class_train, num_classes=2)
multi_model, multi_losses = train_classification_model(X_multi_train, y_multi_train, num_classes=4)

reg_metrics = evaluate_regression(reg_model, X_reg_test, y_reg_test)
class_metrics = evaluate_classification(class_model, X_class_test, y_class_test, num_classes=2)
multi_metrics = evaluate_classification(multi_model, X_multi_test, y_multi_test, num_classes=4)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(reg_losses)
plt.title('Regression Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 3, 2)
plt.plot(class_losses)
plt.title('Binary Classification Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 3, 3)
plt.plot(multi_losses)
plt.title('Multiclass Classification Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

print('\nRegression Metrics:', reg_metrics)
print('\nBinary Classification Metrics:', class_metrics)
print('\nMulticlass Classification Metrics:', multi_metrics)
