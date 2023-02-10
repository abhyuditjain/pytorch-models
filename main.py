from utils.trainer import Trainer
from utils.tester import Tester


def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=20, scheduler=None):
    trainer = Trainer()
    tester = Tester()

    for epoch in range(epochs):
        trainer.train(
            model, train_loader, optimizer, criterion, device, epoch
        )
        if scheduler:
            scheduler.step()
        tester.test(model, test_loader, criterion, device)

    return trainer, tester
