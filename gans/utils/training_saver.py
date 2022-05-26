import torch
import torch.nn as nn
import torch.optim as optim

def save_model(model:nn.Module, optimizer:optim.Optimizer=None, filepath:str="models/chackpoint.mod"):
  checkpoint = {
    "state": model.state_dict()
  }

  if optimizer is not None:
    checkpoint["optim"] = optimizer.state_dict()
    for group in optimizer.param_groups:
      if "lr" in group.keys():
        checkpoint["lr"] = group["lr"]
        break

  torch.save(checkpoint, filepath)

def load_model(model_path:str, model:nn.Module, optimizer:optim.Optimizer, learing_rate:float, device=None):
  if device is None:
    device = torch.device("cpu")

  with open(model_path, "rb") as f:
    checkpoint = torch.load(f, map_location=device)
  model.load_state_dict(checkpoint["state"])

  if optimizer is not None:
    for param_group in optimizer.param_groups:
      param_group["lr"] = learing_rate

  print(f"Loaded {model_path}")

def save_metadata(metadata:dict, filepath:str):
  torch.save(metadata, filepath)

def load_metadata(filepath:str, device=None):
  if device is None:
    device = torch.device("cpu")
  metadata = torch.load(filepath, map_location=device)
  print("Metadata loaded")
  return metadata