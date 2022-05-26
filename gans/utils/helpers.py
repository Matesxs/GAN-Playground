import os

def walk_path(root):
  output_files = []
  for currentpath, folders, files in os.walk(root):
    for file in files:
      output_files.append(os.path.join(currentpath, file))
  return output_files