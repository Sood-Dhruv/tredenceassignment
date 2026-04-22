import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#  Part 1: Prunable Linear Layer 

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=0.01)
        nn.init.normal_(self.gate_scores, mean=0, std=0.1)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return torch.nn.functional.linear(x, pruned_weights, self.bias)


#  Part 2: Model 

class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.relu = nn.ReLU()
        self.fc2 = PrunableLinear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def prunable_layers(self):
        return [self.fc1, self.fc2]


#  Part 3: Data 

def get_loaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


#  Part 4: Sparsity Loss 

def sparsity_loss(model):
    total = 0.0
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.sum()
    return total


#  Part 5: Training Loop 

def train(model, train_loader, lam, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model) 
            loss = ce_loss + lam * sp_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg = running_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg:.4f}")


#  Part 6: Evaluation 

def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def compute_sparsity(model, threshold=1e-2):
    all_gates = []
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores).detach().cpu()
        all_gates.append(gates.view(-1))
    all_gates = torch.cat(all_gates)
    pruned = (all_gates < threshold).float().mean().item()
    return 100.0 * pruned


def get_all_gates(model):
    all_gates = []
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores).detach().cpu()
        all_gates.append(gates.view(-1))
    return torch.cat(all_gates).numpy()


#  Part 7: Run for Multiple Lambdas 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders()

    lambdas = [1e-4, 1e-3, 1e-2]
    epochs = 10
    results = []
    best_model_gates = None
    best_lam = None
    best_sparsity = -1

    for lam in lambdas:
        print(f"\n{'='*50}")
        print(f"Training with lambda = {lam}")
        print('='*50)

        model = SelfPruningNet().to(device)
        train(model, train_loader, lam, epochs, device)

        acc = evaluate(model, test_loader, device)
        sparsity = compute_sparsity(model)
        results.append((lam, acc, sparsity))

        print(f"  -> Accuracy: {acc:.2f}%  |  Sparsity: {sparsity:.2f}%")

        if sparsity > best_sparsity:
            best_sparsity = sparsity
            best_model_gates = get_all_gates(model)
            best_lam = lam

    #  Part 8: Plot

    print(f"\nPlotting gate distribution for best model (lambda={best_lam})")
    plt.figure(figsize=(8, 5))
    plt.hist(best_model_gates, bins=100, color='steelblue', edgecolor='none')
    plt.xlabel('Gate Value')
    plt.ylabel('Count')
    plt.title(f'Gate Value Distribution (λ = {best_lam})')
    plt.tight_layout()
    plt.savefig('gate_distribution.png', dpi=150)
    plt.show()
    print("Saved: gate_distribution.png")

    #  Summary Table

    print(f"\n{'Lambda':<12} {'Accuracy (%)':<16} {'Sparsity (%)'}")
    print('-' * 42)
    for lam, acc, sp in results:
        print(f"{lam:<12} {acc:<16.2f} {sp:.2f}")


if __name__ == '__main__':
    main()