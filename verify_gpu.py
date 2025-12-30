import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch

if torch.cuda.is_available():
    print("✅ GPU is available and PyTorch is set up correctly!")
    print(f"CUDA version: {torch.version.cuda}")
    device_id = torch.cuda.current_device()
    print(f"Device ID: {device_id}")
    print(f"Device name: {torch.cuda.get_device_name(device_id)}")
else:
    print("❌ Warning: PyTorch could not find a GPU. The model will run on the CPU, which will be much slower.")