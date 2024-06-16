
import torch
from torch import Tensor
import torch.nn.functional as F

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))*10

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')*10

def r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    absolute_percentage_error = torch.abs ((y_true - y_pred) / y_true)/100
    # return mean_absolute_percentage_error(y_true, y_pred)
    return torch.mean(absolute_percentage_error)
