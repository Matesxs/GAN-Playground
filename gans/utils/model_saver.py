import torch
import torch.nn as nn
import torch.optim as optim

def save_model(model:nn.Module, optimizer:optim.Optimizer=None, filepath:str="models/chackpoint.mod"):
  checkpoint = {
    "state": model.state_dict()
  }

  if optimizer is not None:
    checkpoint["optim"] = optimizer.state_dict()

  torch.save(checkpoint, filepath)

def load_model(model_path:str, model:nn.Module, optimizer:optim.Optimizer=None, learing_rate:float=None, device=None):
  if device is None:
    device = torch.device("cpu")

  checkpoint = torch.load(model_path, map_location=device)
  model.load_state_dict(checkpoint["state"])

  if optimizer is not None and "optim" in checkpoint.keys():
    for param_group in optimizer.param_groups:
      param_group["lr"] = learing_rate

  print(f"Loaded {model_path}")