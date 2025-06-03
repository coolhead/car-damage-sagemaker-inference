import os
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms

# Define the same model as used during training
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

# Load model from model_dir
def model_fn(model_dir):
    model = SimpleNet()
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Parse incoming image (raw bytes)
def input_fn(request_body, content_type):
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("L")  # grayscale
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Run model prediction
def predict_fn(input_tensor, model):
    with torch.no_grad():
        output = model(input_tensor)
        predicted = output.argmax(dim=1)
        return predicted.item()

# Return result to client
def output_fn(prediction, content_type):
    return str(prediction)

