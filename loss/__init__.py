import torch.nn as nn


def get_loss(name="cross_entropy", device="cuda:0"):
    print(f"Using loss: '{LOSSES[name]}'")
    return LOSSES[name].to(device)


LOSSES = {
    "binary_ce": nn.BCEWithLogitsLoss(),
    "cross_entropy": nn.CrossEntropyLoss()
}
