import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 디바이스 설정 (GPU 있으면 cuda, 없으면 cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2) "직접 만든" 최소 CNN 모델 (Conv2d 사용)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # (N, 1, 28, 28) -> (N, 16, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # (N, 16, 28, 28) -> (N, 16, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # (N, 16, 14, 14) -> (N, 32, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # (N, 32, 14, 14) -> 펼치기 -> (N, 32*14*14)
        self.flatten = nn.Flatten()

        # 최종 분류기: (N, 32*14*14) -> (N, 10)
        self.fc = nn.Linear(32 * 14 * 14, num_classes)

    def forward(self, x):
        print("입력:", x.shape)          # (N, 1, 28, 28)

        x = self.conv1(x)
        print("conv1 후:", x.shape)      # (N, 16, 28, 28)

        x = self.relu(x)
        x = self.pool(x)
        print("pool 후:", x.shape)       # (N, 16, 14, 14)

        x = self.conv2(x)
        print("conv2 후:", x.shape)      # (N, 32, 14, 14)

        x = self.relu(x)

        x = self.flatten(x)
        print("flatten 후:", x.shape)    # (N, 6272)

        x = self.fc(x)
        print("fc 후:", x.shape)         # (N, 10)

        return x


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # 1) forward
        logits = model(x)
        loss = criterion(logits, y)

        # 2) backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def main():
    # 3) MNIST 데이터 준비
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W) -> (1, H, W), 값 범위 [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 표준 정규화(자주 쓰는 값)
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # 4) 모델/손실/옵티마이저
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5) 학습
    epochs = 3
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_acc = evaluate(model, test_loader)

        print(f"[Epoch {epoch}] "
              f"train_loss={train_loss:.4f} "
              f"train_acc={train_acc*100:.2f}% "
              f"test_acc={test_acc*100:.2f}%")

    print("완료!")


if __name__ == "__main__":
    main()
