from models.model9 import TransformerModel
from utils.transforms import CustomResnetTransforms, DefaultTransforms
from utils.dataloader import Cifar10DataLoader
from utils.utils import get_device, show_training_images
from utils.trainer import Trainer
from utils.tester import Tester
from utils.summary import print_summary
from torch_lr_finder import LRFinder
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR


def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format

    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def train_model(
    trainer, tester, NUM_EPOCHS, use_l1=False, scheduler=None, save_best=False
):
    for epoch in range(1, NUM_EPOCHS + 1):
        trainer.train(epoch, scheduler)
        _, test_loss = tester.test()

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)

        if save_best:
            min_val_loss = np.inf
            save_path = "model.pt"
            if test_loss < min_val_loss:
                print(
                    f"Valid loss reduced from {min_val_loss:.5f} to {test_loss:.6f}. checkpoint created at...{save_path}\n"
                )
                save_model(trainer.model, epoch, trainer.optimizer, save_path)
                min_val_loss = test_loss
            else:
                print(f"Valid loss did not inprove from {min_val_loss:.5f}")

        print()

    if scheduler:
        return trainer.model, (
            trainer.train_accuracies,
            trainer.train_losses,
            tester.test_accuracies,
            tester.test_losses,
            trainer.lr_history,
        )
    else:
        return trainer.model, (
            trainer.train_accuracies,
            trainer.train_losses,
            tester.test_accuracies,
            tester.test_losses,
        )


def get_lr(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    end_lr=10,
    num_iter=200,
    step_mode="exp",
    start_lr=None,
    diverge_th=5,
):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        start_lr=start_lr,
        diverge_th=diverge_th,
    )
    lr_finder.plot()
    min_loss = min(lr_finder.history["loss"])
    max_lr = lr_finder.history["lr"][np.argmin(lr_finder.history["loss"], axis=0)]

    print("Min Loss = {}, Max LR = {}".format(min_loss, max_lr))

    # Reset the model and optimizer to initial state
    lr_finder.reset()

    return min_loss, max_lr


def run(epochs: int):
    is_cuda_available, device = get_device()

    print_summary(TransformerModel(), device, input_size=(3, 32, 32))

    model = TransformerModel()
    cifar10 = Cifar10DataLoader(
        DefaultTransforms,
        512,
        is_cuda_available,
        shuffle=False,
    )
    train_loader = cifar10.get_loader(True)
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    criterion = nn.CrossEntropyLoss()

    min_loss, max_lr = get_lr(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        end_lr=0.001,
        num_iter=200,
    )

    cifar10 = Cifar10DataLoader(
        CustomResnetTransforms,
        512,
        is_cuda_available,
    )
    train_loader = cifar10.get_loader(True)
    test_loader = cifar10.get_loader(False)

    show_training_images(train_loader=train_loader, count=20, classes=cifar10.classes)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=5 / epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=1000,
        anneal_strategy="linear",
    )

    trainer = Trainer(model, train_loader, optimizer, criterion, device)
    tester = Tester(model, test_loader, criterion, device)

    train_model(trainer, tester, NUM_EPOCHS=24, scheduler=scheduler)


if __name__ == "__main__":
    run(epochs=24)
