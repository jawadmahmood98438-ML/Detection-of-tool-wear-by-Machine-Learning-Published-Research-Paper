import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV

def train_model(encoder, classifier, train_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)

    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        for x1, x2, labels in train_loader:
            optimizer.zero_grad()
            out1 = encoder(x1).squeeze()
            out2 = encoder(x2).squeeze()
            output = classifier(out1, out2)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()