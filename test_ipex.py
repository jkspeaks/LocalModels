# Courtesy: Claude Sonnet 4

import torch
import intel_extension_for_pytorch as ipex

# Create model and optimizer
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Option 1: Optimize for training (requires optimizer)
model, optimizer = ipex.optimize(model, optimizer=optimizer)
print("IPEX optimization complete (training mode).")

# Option 2: Optimize for inference only (no optimizer needed)
model_inference = torch.nn.Linear(10, 10)
model_inference.eval()  # Set to evaluation mode
model_inference = ipex.optimize(model_inference)
print("IPEX optimization complete (inference mode).")

# Additional verification
print(f"PyTorch version: {torch.__version__}")
print(f"IPEX version: {ipex.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"XPU device count: {torch.xpu.device_count()}")
    print(f"XPU device name: {torch.xpu.get_device_name()}")

# Test with XPU device if available
if torch.xpu.is_available():
    device = torch.device('xpu')
    model = model.to(device)
    x = torch.randn(5, 10).to(device)
    output = model(x)
    print(f"Model output shape: {output.shape}")

    print("Successfully ran model on XPU device!")
