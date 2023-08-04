import torch


class NumCorrect(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, output, target):
        return torch.sum(torch.argmax(output, dim=1) == target).item()
    
def str_to_metric(loss_str, **kwargs):
    loss_str = loss_str.lower()
    if loss_str == "cross_entropy":
        return torch.nn.CrossEntropyLoss(**kwargs)
    elif loss_str == "mse" or loss_str == "mean_squared_error":
        return torch.nn.MSELoss(**kwargs)
    elif loss_str == "accuracy":
        return NumCorrect(**kwargs)
    else:
        raise NotImplementedError(f"Loss {loss_str} not implemented")
