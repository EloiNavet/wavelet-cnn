# train_nets.py
import os
import time
import copy
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
import argparse
import logging

from alexnet import AlexNet
from vgg import VGG

from utils import (
    AverageMeter,
    accuracy,
    resume_from_checkpoint,
    log_to_tensorboard,
    save_model_checkpoint,
    save_final_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def train(
    model: nn.Module,
    trainloader: DataLoader,
    validloader: DataLoader,
    testloader: DataLoader,
    num_classes: int,
    num_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
    wavename: str,
    save_model: bool,
    save_name: str,
    device: torch.device,
) -> nn.Module:
    """
    Train a neural network model.

    Args:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader for the training set.
        validloader (DataLoader): DataLoader for the validation set.
        testloader (DataLoader): DataLoader for the test set.
        num_classes (int): Number of classes in the dataset.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Gamma for the learning rate scheduler.
        wavename (str): Wavelet name for the model.
        save_model (bool): Whether to save the trained model.
        save_name (str): Name for saving the model.
        device (torch.device): Device to run the model on.

    Returns:
        nn.Module: The trained model
    """
    # Create model
    logging.info("==> Building model..")
    # Display the parameters of the model
    logging.info(f"\tModel: {args.model}")
    logging.info(f"\tNumber of classes: {num_classes}")
    logging.info(f"\tNumber of epochs: {num_epochs}")
    logging.info(f"\tLearning rate: {lr}")
    logging.info(f"\tMomentum: {momentum}")
    logging.info(f"\tWeight decay: {weight_decay}")
    logging.info(f"\tWavelet name: {wavename}")
    logging.info(f"\tSave model: {save_model}")
    logging.info(f"\tSave name: {save_name}")
    logging.info(f"\tDevice: {device}")

    summary_writer = SummaryWriter()  # TensorBoard writer

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train the network
    logging.info(f"\n==> Training network (on {device})..")
    best_acc = 0
    since = time.time()

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    if save_model:
        resume_from_checkpoint(model, optimizer, save_name, checkpoint_dir)

    for epoch in range(start_epoch, num_epochs):
        print("\nEpoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device
        )
        valid_loss, valid_acc = test(model, validloader, criterion, device)

        # Log results to TensorBoard
        log_to_tensorboard(
            summary_writer, train_loss, train_acc, valid_loss, valid_acc, epoch
        )

        # Display progress
        logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logging.info(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")
        logging.info(f"Learning Rate: {scheduler.get_last_lr()[0]}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

        if save_model:
            save_model_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                save_name,
                checkpoint_dir,
            )

    time_elapsed = time.time() - since
    logging.info(
        f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    logging.info(f"Best validation accuracy: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Evaluate on the test set
    test_loss, test_acc = test(model, testloader, criterion, device)
    print("\nTest Loss: {:.4f} | Test Acc: {:.4f}".format(test_loss, test_acc))

    # Save the final trained model
    if save_model:
        save_final_model(
            model,
            optimizer,
            num_epochs,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
            save_name,
            checkpoint_dir,
        )

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): Device to run the model on.

    Returns:
        tuple[float, float]: Average loss and accuracy for the epoch.
    """
    model.train()
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    for inputs, targets in tqdm(dataloader, desc="Training", unit="batch", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc = accuracy(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.item(), inputs.size(0))
        epoch_acc.update(acc.item(), inputs.size(0))

    return epoch_loss.avg, epoch_acc.avg


def test(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the model on validation or test sets.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): DataLoader for the validation or test set.
        criterion (nn.Module): The loss function.
        device (torch.device): Device to run the model on.

    Returns:
        tuple[float, float]: Average loss and accuracy for the dataset.
    """
    model.eval()
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    with torch.no_grad():
        for inputs, targets in tqdm(
            dataloader, desc="Testing", unit="batch", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc = accuracy(outputs, targets)

            epoch_loss.update(loss.item(), inputs.size(0))
            epoch_acc.update(acc.item(), inputs.size(0))

    return epoch_loss.avg, epoch_acc.avg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CNN models on Tiny-ImageNet-C dataset"
    )
    parser.add_argument(
        "--model", type=str, default="alexnet", help="Model name (alexnet or vgg)"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--validation_split", type=float, default=0.15, help="Validation split ratio"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--wavename", type=str, default="haar", help="Wavelet name")
    parser.add_argument(
        "--save_model", action="store_true", help="Save the trained model"
    )
    parser.add_argument(
        "--save_name", type=str, default="alexnet", help="Name for saving the model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=30,
        help="Step size for the learning rate scheduler",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Image transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    imagenet_dataset_path = os.path.join(
        os.getcwd(), "data", "Tiny-ImageNet-C", "brightness", "1"
    )
    imagenet_dataset = ImageFolder(root=imagenet_dataset_path, transform=transform)

    # Dataset splits
    total_size = len(imagenet_dataset)
    validation_size = int(args.validation_split * total_size)
    test_size = int(args.validation_split * total_size)
    train_size = total_size - validation_size - test_size

    train_dataset, validation_dataset, test_dataset = random_split(
        imagenet_dataset,
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    num_classes = len(imagenet_dataset.classes)
    logging.info(f"Number of classes: {num_classes}")

    device = torch.device(args.device)
    if args.model == "alexnet":
        model = AlexNet(num_classes=num_classes, wavename=args.wavename)
    elif args.model == "vgg":
        model = VGG(num_classes=num_classes, wavename=args.wavename)
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    model = model.to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Start training
    train(
        model,
        train_loader,
        validation_loader,
        test_loader,
        num_classes,
        args.num_epochs,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.step_size,
        args.gamma,
        args.wavename,
        args.save_model,
        args.save_name,
        device,
    )
