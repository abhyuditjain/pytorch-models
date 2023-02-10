import torch
import torch.optim as optim
import torch.nn as nn
from utils.dataloader import Cifar10DataLoader
from models import ResNet18
from utils.trainer import Trainer
from utils.tester import Tester
from utils.summary import print_summary





def main():
    is_cuda_available = torch.cuda.is_available()
    print("Is GPU available?", is_cuda_available)
    device = torch.device("cuda" if is_cuda_available else "cpu")
    cifar10 = Cifar10DataLoader(is_cuda_available=is_cuda_available)
    train_loader = cifar10.get_loader(True)
    test_loader = cifar10.get_loader(False)
    model = ResNet18().to(device=device)

    print_summary(model)

    trainer = Trainer()
    tester = Tester()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 200

    for epoch in range(EPOCHS):
        trainer.train(
            model, train_loader, optimizer, criterion, device, epoch
        )
        tester.test(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
