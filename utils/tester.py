import torch


class Tester:
    def __init__(self) -> None:
        self.test_losses = []
        self.test_accuracies = []

    def test(self, model, dataloader, criterion, device):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)

                test_loss += loss.item()

                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(dataloader.dataset),
                100.0 * correct / len(dataloader.dataset),
            )
        )

        self.test_accuracies.append(100.0 * correct / len(dataloader.dataset))
