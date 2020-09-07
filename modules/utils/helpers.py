import os
import subprocess
from colorama import Fore

def get_paths_of_files_from_path(path, only_files:bool=False):
  if not os.path.exists(path): return None
  content = [os.path.join(path, file_name) for file_name in os.listdir(path)]
  if only_files:
    content = [x for x in content if os.path.isfile(x)]
  return content

def time_to_format(timestamp):
  # Helper vars:
  MINUTE = 60
  HOUR = MINUTE * 60
  DAY = HOUR * 24
  MONTH = DAY * 30

  # Get the days, hours, etc:
  months = int(timestamp / MONTH)
  days = int((timestamp % MONTH) / DAY)
  hours = int((timestamp % DAY) / HOUR)
  minutes = int((timestamp % HOUR) / MINUTE)
  seconds = int(timestamp % MINUTE)

  # Build up the pretty string (like this: "N days, N hours, N minutes, N seconds")
  string = ""
  if months > 0:
    string += str(months) + " " + (months == 1 and "month" or "months") + ", "
  if days > 0:
    string += str(days) + " " + (days == 1 and "day" or "days") + ", "
  if len(string) > 0 or hours > 0:
    string += str(hours) + " " + (hours == 1 and "hour" or "hours") + ", "
  if len(string) > 0 or minutes > 0:
    string += str(minutes) + " " + (minutes == 1 and "minute" or "minutes") + ", "
  string += str(seconds) + " " + (seconds == 1 and "second" or "seconds")

  return string

# Calculate start image size based on final image size and number of upscales
def count_upscaling_start_size(target_image_shape: tuple, num_of_upscales: int):
  upsc = (target_image_shape[0] // (2 ** num_of_upscales), target_image_shape[1] // (2 ** num_of_upscales), target_image_shape[2])
  if upsc[0] < 1 or upsc[1] < 1: raise Exception(f"Invalid upscale start size! ({upsc})")
  return upsc

def start_tensorboard(data_path):
  try:
    if os.path.exists("./venv/Scripts"):
      return subprocess.Popen(f"./venv/Scripts/python.exe -m tensorboard.main --logdir {data_path} --samples_per_plugin=images=200 --port 6006", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
      return subprocess.Popen(f"python -m tensorboard.main --logdir {data_path} --samples_per_plugin=images=200 --port 6006", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  except:
    try:
      return subprocess.Popen(f"tensorboard --logdir {data_path} --samples_per_plugin=images=200 --port 6006", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
      print(Fore.YELLOW + "Cant start tensorboard thread, check helper function in helper module start_tensorboard in modules/utils and change way of starting tensorboard thread according to your system" + Fore.RESET)
      return None