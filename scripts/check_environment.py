"""Print the Python/PyTorch environment before training or GUI use."""
import platform
import sys

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
try:
    import torch
    import torchvision
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Could not import PyTorch/torchvision:", repr(exc))
