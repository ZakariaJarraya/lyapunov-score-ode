# Fix the path before importing run_lib
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import torch

import torch.nn as nn
import torch.optim as optim
import sys
import os
from pathlib import Path
from my_datasets import datasets
from tqdm import tqdm



from models.classifier import classifier 






def train_epoch(model, train_loader, criterion, learning_rate, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())




def evaluate(model, test_loader, i):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    for inputs, labels in test_loader:
      with torch.no_grad():
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.squeeze().long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print (f"epoch {i}: accuracy = {accuracy}")    






def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = classifier.get_classifier(args.model)  
    model = classifier.adapt_model(model, args.dataset)
    model = model.to(device)
    print(f"--- Running on device: {device} ---")
    if device.type == 'cuda':
        print(f"--- GPU Name: {torch.cuda.get_device_name(0)} ---")
    else:
        print("--- WARNING: CUDA NOT FOUND. TRAINING WILL BE SLOW. ---")
    train_loader, test_loader, _, _ = datasets.get_dataloaders(args.dataset, args.batch_size)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(args.n_epoch):
        train_epoch(model, train_loader, criterion, learning_rate, optimizer)
        evaluate(model, test_loader, i)
    checkpoint = {
    'epoch': args.n_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, args.dataset +'_' + args.model +'_checkpoint.pth')   


