r"""
This document is a simpel Neural Network that classify some
vectors that are in a .csv. The vectors that we use are
the graph representation achieved with the Weisfeirel-Lehman
representation with one iteration. 

We do this to compare how actually works 
"""

import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# === 1. Cargar y procesar el .csv ===
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Convertir columna "Graphs" de string a lista
    df["Graphs"] = df["Graphs"].apply(ast.literal_eval)

    X = df["Graphs"].tolist()
    y = df["Kingdom"].tolist()

    # Codificar las etiquetas si no son numéricas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y_encoded, dtype=torch.long), label_encoder

# === 2. Red neuronal flexible ===
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(SimpleNN, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# === 3. Entrenamiento y evaluación ===
def train_model(X, y, hidden_sizes=[64, 32], epochs=20, batch_size=32, lr=0.001):
    num_classes = len(torch.unique(y))
    input_size = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    model = SimpleNN(input_size, hidden_sizes, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return model

x, y, l = load_data(f'Weisfeirel-Lehman\\data\\data_base_n2.csv')
train_model(x,y,[128,64,64], epochs=100)