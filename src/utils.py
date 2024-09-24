# utils.py
import os
import torch


class AverageMeter:
    """Computes and stores the average and current values for tracking metrics like loss and accuracy."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> torch.Tensor:
    """
    Compute top-k accuracy for classification models.

    Args:
        output (torch.Tensor): The model output predictions.
        target (torch.Tensor): The true labels.
        topk (int): The top-k accuracy to compute.

    Returns:
        torch.Tensor: The top-k accuracy.
    """
    top_indices = torch.topk(output.data, topk)[1].t()
    match = top_indices.eq(target.view(1, -1).expand_as(top_indices))
    acc = match.view(-1).float().mean() * topk
    return acc


def resume_from_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    save_name: str,
    checkpoint_dir: str,
) -> int:
    """
    Resumes training from the latest checkpoint.

    Args:
        model (torch.nn.Module): The model whose weights need to be restored.
        optimizer (torch.optim.Optimizer): The optimizer whose state needs to be restored.
        save_name (str): The name used to save the checkpoints.
        checkpoint_dir (str): The directory where the checkpoint files are stored.

    Returns:
        int: The epoch number to resume training from.
    """

    # List all checkpoint files in the directory that match the save_name pattern
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith(save_name) and f.endswith(".pth")
    ]

    if len(checkpoint_files) == 0:
        # No checkpoint found, so training starts from scratch
        print(
            f"No checkpoint found in directory {checkpoint_dir}. Training will start from scratch."
        )
        return 0

    # Find the latest checkpoint by extracting the epoch number from the file names
    checkpoint_epochs = [
        int(f.split("_")[1].split(".")[0].replace("epoch", ""))
        for f in checkpoint_files
    ]
    latest_epoch = max(checkpoint_epochs)

    # Load the latest checkpoint file
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{save_name}_epoch{latest_epoch}.pth"
    )
    checkpoint = torch.load(checkpoint_path)

    # Load model and optimizer states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Resumed training from checkpoint: {checkpoint_path} (Epoch {latest_epoch})")

    return latest_epoch


def log_to_tensorboard(
    writer: torch.utils.tensorboard.SummaryWriter,
    train_loss: float,
    train_acc: float,
    valid_loss: float,
    valid_acc: float,
    epoch: int,
):
    """Log training metrics to TensorBoard."""
    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_acc, epoch)
    writer.add_scalar("Validation/Loss", valid_loss, epoch)
    writer.add_scalar("Validation/Accuracy", valid_acc, epoch)


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    train_acc: float,
    valid_loss: float,
    valid_acc: float,
    save_name: str,
    checkpoint_dir: str,
):
    """
    Saves a checkpoint during training.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): The current epoch number.
        train_loss (float): Training loss at the current epoch.
        train_acc (float): Training accuracy at the current epoch.
        valid_loss (float): Validation loss at the current epoch.
        valid_acc (float): Validation accuracy at the current epoch.
        save_name (str): The prefix name to use for saving the model.
        checkpoint_dir (str): The directory to save the checkpoint in.
    """

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint file path
    checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_epoch{epoch}.pth")

    # Save the model, optimizer, and additional information
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved at: {checkpoint_path}")


def save_final_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    train_loss: float,
    train_acc: float,
    valid_loss: float,
    valid_acc: float,
    test_loss: float,
    test_acc: float,
    save_name: str,
    checkpoint_dir: str,
):
    """
    Saves the final model after training is complete.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        num_epochs (int): The total number of epochs.
        train_loss (float): Final training loss.
        train_acc (float): Final training accuracy.
        valid_loss (float): Final validation loss.
        valid_acc (float): Final validation accuracy.
        test_loss (float): Final test loss.
        test_acc (float): Final test accuracy.
        save_name (str): The prefix name for saving the final model.
        checkpoint_dir (str): The directory to save the model in.
    """

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the final model path
    final_model_path = os.path.join(checkpoint_dir, f"{save_name}_final.pth")

    # Save the model, optimizer, and additional information
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
        final_model_path,
    )

    print(f"Final model saved at: {final_model_path}")
