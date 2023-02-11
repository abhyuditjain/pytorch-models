from utils.trainer import Trainer
from utils.tester import Tester


def get_train_runner(model, train_loader, optimizer, criterion, device):
    trainer = Trainer(model, train_loader, optimizer, criterion, device)
    return trainer


def get_test_runner(model, test_loader, criterion, device):
    tester = Tester(model, test_loader, criterion, device)
    return tester


def train_model(trainer, tester, epochs=20, scheduler=None):
    for epoch in range(epochs):
        trainer.train(epoch)
        if scheduler:
            scheduler.step()
        tester.test()
