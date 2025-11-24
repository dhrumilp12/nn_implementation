""" How to run:
            python base_experiment.py
"""

from collections import Counter

import numpy as np
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from custom_modules import Linear, CrossEntropyLoss, Sigmoid
from torch.nn import Module

# Define the neural network model
class FashionNet(Module):
    """
    Single-hidden-layer NN:
        784 -> 256 -> 10
    Hidden activation: Sigmoid
    Output: logits (softmax is inside CrossEntropyLoss)
    """

    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.act = Sigmoid()
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (N, 1, 28, 28)
        x = x.view(x.size(0), -1)  # flatten to (N, 784)
        z = self.fc1(x)
        z = self.act(z)
        logits = self.fc2(z)
        return logits

# Evaluation function
def evaluate(model, criterion, data_loader, device):
    """
    Compute average loss and accuracy over a dataset.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# Training function for one epoch
def train_one_epoch(model, criterion, optimizer, train_loader, device):
    """
    Perform one training epoch with SGD (no shuffling).
    """
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# Main experiment runner
def run_experiment(
    trainset,
    testset,
    batch_size=1,
    num_epochs=15,
    lr=0.01,
    hidden_dim=256,
    device="cpu",
):
    """
    Generic experiment runner: trains model and prints per-epoch stats.
    Returns history dict with losses and accuracies.
    """
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = FashionNet(hidden_dim=hidden_dim).to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, criterion, optimizer, train_loader, device)

        # IMPORTANT per instructions: compute training loss over full train set
        train_loss, _ = evaluate(model, criterion, train_loader, device)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    return model, history

# Utility functions for Q4 and Q5
def get_all_preds_labels(model, data_loader, device):
    """
    Utility to collect all predictions and labels (for confusion matrix, etc.)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels

# Confusion matrix plotting
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# First misclassified per class
def show_first_misclassified_per_class(model, data_loader, class_names, device):
    """
    For each class, show the first misclassified test image.
    """
    model.eval()
    found = {}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            for i in range(labels.size(0)):
                true_lbl = labels[i].item()
                pred_lbl = preds[i].item()
                if true_lbl not in found and true_lbl != pred_lbl:
                    found[true_lbl] = (images[i].cpu(), pred_lbl)
            if len(found) == len(class_names):
                break

    num_classes = len(class_names)
    plt.figure(figsize=(12, 6))
    for c in range(num_classes):
        if c in found:
            img, pred_lbl = found[c]
            plt.subplot(2, 5, c + 1)
            plt.imshow(img.squeeze(0), cmap="gray")
            plt.axis("off")
            plt.title(f"True: {class_names[c]}\nPred: {class_names[pred_lbl]}")
    plt.tight_layout()
    plt.savefig("q5_first_misclassified_per_class.png")
    plt.show()

# Main script to run experiments
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Fashion-MNIST
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # ===== Q1 & Q2: SGD (batch_size = 1), 15 epochs =====
    # model_sgd, history_sgd = run_experiment(
    #    trainset=trainset,
    #    testset=testset,
    #    batch_size=1,        # SGD
    #   num_epochs=15,
    #   lr=0.01,
    #   hidden_dim=256,
    #   device=device,
    # )

    # After this run finishes, you can copy from the console:
    #  - average test loss per epoch  (Q1)
    #  - test accuracy per epoch      (Q2)

    # save weights if want to reuse for later parts
    # torch.save(model_sgd.state_dict(), "fashion_mnist_sgd.pt")



    
    # ===== Q3: mini-batch GD, batch_size = 5, 50 epochs =====
    model_b5, history_b5 = run_experiment(
        trainset=trainset,
        testset=testset,
        batch_size=5,       # mini-batch size = 5
        num_epochs=50,
        lr=0.01,
        hidden_dim=256,
        device=device,
    )

    # Final values for Q3 (epoch 50)
    final_train_loss = history_b5["train_loss"][-1]
    final_test_acc = history_b5["test_acc"][-1]
    print(f"\n=== Q3 Results (batch=5, epoch=50) ===")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")


    # ===== Q4: Confusion matrices for model trained with batch=5, 50 epochs =====

    train_loader_full = DataLoader(trainset, batch_size=128, shuffle=False)
    test_loader_full = DataLoader(testset, batch_size=128, shuffle=False)

    train_preds, train_labels = get_all_preds_labels(model_b5, train_loader_full, device)
    test_preds, test_labels = get_all_preds_labels(model_b5, test_loader_full, device)

    num_classes = 10
    train_cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    test_cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Fill confusion matrices: rows = true, cols = predicted
    for t, p in zip(train_labels, train_preds):
        train_cm[t, p] += 1
    for t, p in zip(test_labels, test_preds):
        test_cm[t, p] += 1

    class_names = trainset.classes  # ["T-shirt/top", "Trouser", ..., "Ankle boot"]

    plot_confusion_matrix(train_cm, class_names, "Train Confusion Matrix (batch=5, 50 epochs)")
    plot_confusion_matrix(test_cm, class_names, "Test Confusion Matrix (batch=5, 50 epochs)")

    
    # Q5: first misclassified per class on test:
    show_first_misclassified_per_class(
        model_b5, test_loader_full, class_names, device
    )

    # ===== Q6: Batch size experiments [10, 50, 100], 50 epochs =====
    batch_sizes = [10, 50, 100]
    num_epochs = 50
    lr = 0.01

    histories_bs = {}

    for bs in batch_sizes:
        print(f"\n=== Running batch size {bs} for {num_epochs} epochs ===")
        _, hist = run_experiment(
            trainset=trainset,
            testset=testset,
            batch_size=bs,
            num_epochs=num_epochs,
            lr=lr,
            hidden_dim=256,
            device=device,
        )
        histories_bs[bs] = hist

    # Plot epoch vs TRAIN loss for each batch size
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    for bs in batch_sizes:
        plt.plot(epochs, histories_bs[bs]["train_loss"], label=f"batch={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch for Different Batch Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("q6_train_loss_vs_epoch.png")
    plt.show()

    # Plot epoch vs TEST loss for each batch size
    plt.figure(figsize=(8, 6))
    for bs in batch_sizes:
        plt.plot(epochs, histories_bs[bs]["test_loss"], label=f"batch={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs Epoch for Different Batch Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig("q6_test_loss_vs_epoch.png")
    plt.show()
