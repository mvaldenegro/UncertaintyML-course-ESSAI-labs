import torch
from torch_helpers.metrics import str_to_metric



def train(model, train_loader, epochs, optimizer, loss_fn, device, metrics=None):
    if metrics is None:
        metrics = []
    
    if isinstance(loss_fn, str):
        loss_fn = str_to_metric(loss_fn, reduction="sum")
    metrics = [str_to_metric(metric, reduction="sum") for metric in metrics]
    model.train()

    for epoch in range(epochs):
        loss_accumulator = 0.0
        metric_accumulators = [0.0 for _ in metrics]
        n = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            n += data.shape[0]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss_accumulator += loss.item()
            (loss/n).backward()
            with torch.no_grad():
                for metric, metric_accumulator in zip(metrics, metric_accumulators):
                    metric_accumulator += metric(output, target)
            optimizer.step()
        
        metric_strings = [f"{metric.__class__.__name__}: {metric_accumulator/n}" for metric, metric_accumulator in zip(metrics, metric_accumulators)]
        print(f"Epoch {epoch+1}: loss {loss_accumulator/n} - " + " - ".join(metric_strings))
    