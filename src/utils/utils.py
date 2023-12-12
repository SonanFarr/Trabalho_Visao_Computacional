import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import glob as gb
import os
from sklearn.metrics import confusion_matrix

def train(dataloader, model, loss_fn, optimizer, device):
    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)

    num_batches = len(dataloader)
    # Indica que o modelo está em processo de treinamento
    model.train()

    # Define a loss total do treinamento
    totalLoss = 0

    # Itera sobre os lotes
    for batch, (X, y) in enumerate(dataloader):
        # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
        X, y = X.to(device), y.to(device)

        # Faz a predição para os valores atuais dos parâmetros
        pred = model(X)

        # Estima o valor da função de perda
        loss = loss_fn(pred, y)

        # Incrementa a loss total
        totalLoss += loss

        # Backpropagation

        # Limpa os gradientes
        optimizer.zero_grad()

        # Estima os gradientes
        loss.backward()

        # Atualiza os pesos da rede
        optimizer.step()

        # LOG: A cada 100 lotes (iterações) mostra a perda
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
    totalLoss /= num_batches
    return totalLoss

def test(dataloader, model, loss_fn, device):
    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)

    # Obtém o número de lotes (iterações)
    num_batches = len(dataloader)

    # Indica que o modelo está em processo de teste
    model.eval()

    # Inicializa a perda de teste e a quantidade de acertos com 0
    test_loss, correct = 0, 0

    # Desabilita o cálculo do gradiente
    with torch.no_grad():
        # Itera sobre o conjunto de teste
        for X, y in dataloader:
            # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
            X, y = X.to(device), y.to(device)
            # Realiza a predição
            pred = model(X)

            # Calcula a perda
            test_loss += loss_fn(pred, y).item()
            # Verifica se a predição foi correta
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Determina a perda média e a proporção de acertos
    test_loss /= num_batches
    correct /= size
    # LOG: mostra a acurácia e a perda
    return (100*correct), test_loss

def test_for_class(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    pred_list = []
    label_list = []

    acc = [0 for c in range(10)]
    y_list = [0 for c in range(10)]

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            pred_list.append(pred.argmax(1).cpu().detach().numpy())
            label_list.append(y.cpu().detach().numpy())

            for c in range(10):
                acc[c] += (((pred.argmax(1) == y) * (y == c)).type(torch.float).sum().item())
                y_list[c] += y.tolist().count(c)
    for k in range(10):
        acc[k] /= y_list[k]

    test_loss /= num_batches
    return pred_list, label_list, acc, test_loss  

def test_new_metrics(dataloader, model, loss_fn, device): 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    pred_list = []
    label_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            pred_list.append(pred.argmax(1).cpu().detach().numpy())
            label_list.append(y.cpu().detach().numpy())

    test_loss /= num_batches
    correct /= size
    
    return pred_list, label_list, (100*correct), test_loss