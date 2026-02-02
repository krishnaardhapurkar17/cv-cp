import pathlib

import torch

# This is a temporary trick to fool the unpickler.
# It tells Python to create a WindowsPath object when it sees a PosixPath object in the file.
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the problematic model file, explicitly allowing it to load more than just weights.
print("Loading the original model file...")
model = torch.load("best.pt", map_location=torch.device("cpu"), weights_only=False)

# Restore the original PosixPath class
pathlib.PosixPath = temp

# Re-save the model. This time it will be saved in a compatible format.
print("Re-saving the model in a compatible format...")
torch.save(model, "best_fixed.pt")

print("\nDone! A new file named 'best_fixed.pt' has been created. Use this file for detection.")
