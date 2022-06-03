import torch.nn as nn

def get_layer_outputs(model, inputs):
  layer_outputs = []

  def get_activation():
    def hook(model, input, output):
      layer_outputs.append(output.detach())
    return hook

  modules = model.named_modules()
  for name, layer in modules:
    if type(layer) in (nn.Conv2d, nn.ConvTranspose2d):
      layer.register_forward_hook(get_activation())

  model(*inputs)

  return layer_outputs