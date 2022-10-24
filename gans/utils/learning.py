from torch.optim import lr_scheduler

def get_linear_decay_scheduler(optimizer, steps_before_decay, decay_steps, initial_steps=0):
  def lambda_rule(step):
    lr_l = max(1.0 - max(0, step - steps_before_decay + initial_steps) / float(decay_steps), 0.0)
    return lr_l

  return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
