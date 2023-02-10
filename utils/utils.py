import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import torch


def show_grad_cam(model, device, images, predictions, use_cuda=True):

    target_layers = [model.layer3[-2]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    fig = plt.figure(figsize=(40, 40))

    for i in range(len(images)):
        input_tensor = images[i].unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(predictions[i])]
        rgb_img = denormalize(images[i].cpu().numpy().squeeze())
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        fig.add_subplot(2, 10, i + 1)
        plt.imshow(rgb_img, cmap="gray")
        fig.add_subplot(2, 10, i + 1 + 10)
        plt.imshow(visualization, cmap="gray")
    plt.tight_layout()
    plt.show()


def denormalize(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))


def show_training_images(train_loader, count, classes):
    images, labels = next(iter(train_loader))
    images = images[0:count]
    labels = labels[0:count]

    fig = plt.figure(figsize=(20, 10))
    for i in range(count):
        sub = fig.add_subplot(count/5, 5, i+1)
        npimg = denormalize(images[i].cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        sub.set_title("Correct class: {}".format(classes[labels[i]]))
    plt.tight_layout()
    plt.show()


def show_misclassified_images(model, test_loader, classes, device):
    model.eval()

    misclassified_images = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append({'image': data[i], 'predicted_class': pred[i], 'correct_class': target[i]})

    # Plot the misclassified images
    fig = plt.figure(figsize=(20, 10))
    for i in range(20):
        sub = fig.add_subplot(4, 5, i+1)
        misclassified_image = misclassified_images[i]
        npimg = denormalize(misclassified_image['image'].cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        correct = classes[misclassified_image['correct_class']]
        predicted = classes[misclassified_image['predicted_class']]
        sub.set_title("Correct class: {}\nPredicted class: {}".format(correct, predicted))
    plt.tight_layout()
    plt.show()

    return torch.from_numpy(np.array([mi['image'].cpu().numpy().squeeze() for mi in misclassified_images])), [mi['predicted_class'] for mi in misclassified_images]