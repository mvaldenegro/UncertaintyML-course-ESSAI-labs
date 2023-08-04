from typing import Iterable, Union
import torch

def str_to_aggregation_fn(aggregation_str):
    if aggregation_str == "mean":
        return torch.mean()

def predict_single(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        output = []
        for data, target in dataloader:
            data = data.to(device)
            output.append(model(data))
    return torch.cat(output, dim=0)

class StochasticPredictor(torch.nn.Module):
    def __init__(self, model, num_samples):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
    
    def forward(self, x):
        return torch.stack([self.model(x) for _ in range(self.num_samples)], dim=0)
    
    def _predict(self, dataloader, device):
        '''
            Used to get predictions for a dataloader over multiple samples, as defined by forward.
            Not meant to be called directly, use predict instead.
        '''
        self.eval()
        with torch.no_grad():
            output = []
            for data, target in dataloader:
                data = data.to(device)
                pred = self(data)
                output.append(pred)
                
        return torch.cat(output, dim=1)
    
    def predict(self, dataloader, device, aggregations:Union[None, Iterable]):
        '''
            Returns a tuple of predictions and uncertainties. Tune aggregations accordingly.

            Params:
                dataloader: torch.utils.data.DataLoader
                device: torch.device
                aggregations: Iterable of torch functions to aggregate over the samples.
                              If None, returns the predictions as is.
                              If dict, Keys are the torch functions, values are the dimensions 
                              to aggregate over. If list or similar, aggregates over the first
                              dimension (samples).
        '''
        '''
            Returns a tuple of predictions and uncertainties. Tune aggregations accordingly.

            Params:
                dataloader: torch.utils.data.DataLoader
                device: torch.device
                aggregations: dictionary of torch functions to aggregate over the samples.
                              Keys are the torch functions, values are the dimensions to aggregate over.
        '''
        predictions = self._predict(dataloader, device)
        if aggregations is None:
            return predictions

        if not isinstance(aggregations, dict):
            aggregations = {aggregation_fn: 0 for aggregation_fn in aggregations}

        return tuple([aggregation_fn(predictions, dim=dim) for aggregation_fn, dim in aggregations.items()])


class StochasticRegressor(StochasticPredictor):
    '''
        A wrapper around a torch model performing regression that allows for stochastic predictions.
        The predictions are aggregated for mean and std.
    '''
    def predict(self, dataloader, device):
        '''
            Returns a tuple of mean predictions and uncertainties.
        '''
        return super().predict(dataloader, device, aggregations={torch.mean: 0, torch.std: 0})


class EnsemblePredictor(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
    
    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=0)
    
    def predict(self, dataloader, device, aggregations:Union[None, Iterable]):
        '''
            Returns a tuple of predictions and uncertainties. Tune aggregations accordingly.

            Params:
                dataloader: torch.utils.data.DataLoader
                device: torch.device
                aggregations: Iterable of torch functions to aggregate over the samples.
                              If None, returns the predictions as is.
                              If dict, Keys are the torch functions, values are the dimensions 
                              to aggregate over. If list or similar, aggregates over the first
                              dimension (samples).
        '''
        predictions = self._predict(dataloader, device)
        if aggregations is None:
            return predictions

        if not isinstance(aggregations, dict):
            aggregations = {aggregation_fn: 0 for aggregation_fn in aggregations}

        return tuple([aggregation_fn(predictions, dim=dim) for aggregation_fn, dim in aggregations.items()])

    def _predict(self, dataloader, device):
        '''
            Used to get predictions for a dataloader over multiple samples, as defined by forward.
            Not meant to be called directly, use predict instead.
        '''
        self.eval()
        with torch.no_grad():
            output = []
            for data, target in dataloader:
                data = data.to(device)
                pred = self(data)
                output.append(pred)
                
        return torch.cat(output, dim=1)

class EnsemblePredictorWithUncertainty(EnsemblePredictor):
    '''
        A wrapper around a torch model performing regression that allows for stochastic predictions.
        The underlying models are two-headed regressor neural network estimating mean and variance.
    '''
    def forward(self, x):
        mean, var = [], []
        for model in self.models:
            member_mean, member_var = model(x)
            mean.append(member_mean)
            var.append(member_var)
        return torch.stack(mean, dim=0), torch.stack(var, dim=0)

    def _predict(self, dataloader, device):
        '''
            Used to get predictions for a dataloader over multiple samples, as defined by forward.
            Not meant to be called directly, use predict instead.
        '''
        self.eval()
        with torch.no_grad():
            output_mean = []
            output_var = []
            for data, target in dataloader:
                data = data.to(device)
                mean, var = self(data)
                output_mean.append(mean)
                output_var.append(var)
                
        return torch.cat(output_mean, dim=1), torch.cat(output_var, dim=1)

class EnsembleRegressorWithUncertainty(EnsemblePredictorWithUncertainty):
    def predict(self, dataloader, device):
        '''
            Returns a tuple of mean predictions and uncertainties.
        '''
        means, vars = self._predict(dataloader, device)
        mu = torch.mean(means, dim=0)
        sigma2 = torch.mean(vars + means**2, dim=0) - mu**2
        sigma2.clip_(min=0)
        return mu, sigma2.sqrt()