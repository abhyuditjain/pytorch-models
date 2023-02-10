from tqdm import tqdm


class Trainer:
    def __init__(self) -> None:
        self.train_losses = []
        self.train_accuracies = []
        self.epoch_train_accuracies = []

    def train(self, model, dataloader, optimizer, criterion, device, epoch):
        model.train()
        correct = 0
        processed = 0

        pbar = tqdm(dataloader)

        for batch_id, (inputs, targets) in enumerate(pbar):
            # transfer to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize gradients to 0
            optimizer.zero_grad()

            # Prediction
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            processed += len(inputs)

            pbar.set_description(
                desc=f"EPOCH = {epoch} | LR = {optimizer.param_groups[0]['lr']} | Loss = {loss.item():3.2f} | Batch = {batch_id} | Accuracy = {100*correct/processed:0.2f}"
            )
            self.train_accuracies.append(100 * correct / processed)

        # After all the batches are done, append accuracy for epoch
        self.epoch_train_accuracies.append(100 * correct / processed)
        return 100 * correct / processed
