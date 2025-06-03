# 🚗 Car Damage Detection MLOps Pipeline (AWS SageMaker + PyTorch + GitHub Actions)

This project demonstrates an end-to-end **MLOps pipeline** to detect car damage using a PyTorch model deployed on **AWS SageMaker**. The pipeline involves annotation, training, packaging, deployment, and real-time inference using SageMaker endpoints.

---

## 🛠️ Project Structure

```
car-damage-pipeline/
├── model_dir/
│   ├── model.pth               # Trained PyTorch model
│   ├── inference.py           # Inference script for SageMaker
│   └── model.tar.gz           # Deployment package for SageMaker
├── pipeline/
│   ├── train.py               # (Optional) Training script
│   ├── evaluate.py            # (Optional) Evaluation script
│   └── preprocessing.py       # (Optional) Preprocessing script
├── predict.py                 # Script to invoke the SageMaker endpoint
├── config/config.yaml         # Configuration file (optional)
├── .github/workflows/
│   └── deploy.yml             # GitHub Actions CI/CD pipeline (To Be Added)
└── README.md                  # Project documentation
```

---

## ✅ Features

* ✅ Annotate car damage images with **Roboflow**
* ✅ Train a **PyTorch** model (simple net in this case)
* ✅ Package model + inference script into `model.tar.gz`
* ✅ Upload to **S3** and deploy on **SageMaker**
* ✅ Inference with a `predict.py` client

---

## 🔁 Inference Pipeline Flow

1. `predict.py` loads a test image as raw bytes.
2. Sends it to the SageMaker endpoint: `car-damage-final-infer-endpoint`
3. Inference logic is defined in `inference.py`
4. Returns predicted class.

---

## 🔍 Inference Script: `inference.py`

```python
import os
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

def model_fn(model_dir):
    model = SimpleNet()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=torch.device("cpu")))
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("L")
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_tensor, model):
    with torch.no_grad():
        output = model(input_tensor)
        return output.argmax(dim=1).item()

def output_fn(prediction, content_type):
    return str(prediction)
```

---

## 🧠 Predict Script: `predict.py`

```python
import boto3

endpoint_name = "car-damage-final-infer-endpoint"
region = "ap-south-1"
image_path = "test_image.jpg"

with open(image_path, "rb") as f:
    payload = f.read()

client = boto3.client("sagemaker-runtime", region_name=region)
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/x-image",
    Body=payload
)

print("🔮 Prediction:", response["Body"].read().decode("utf-8"))
```

---

## 📦 Packaging Model for SageMaker

```bash
cd model_dir
rm -f model.tar.gz

tar -czvf model.tar.gz inference.py model.pth
aws s3 cp model.tar.gz s3://car-damage-dataset-bucket/model/model.tar.gz --region ap-south-1
```

---

## 🚀 Create SageMaker Model and Endpoint

```bash
# Step 1: Create Model
aws sagemaker create-model \
  --model-name car-damage-final-infer-model \
  --primary-container Image="763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38",ModelDataUrl="s3://car-damage-dataset-bucket/model/model.tar.gz" \
  --execution-role-arn arn:aws:iam::<account-id>:role/AmazonSageMaker-ExecutionRole-CarDamage \
  --region ap-south-1

# Step 2: Create Endpoint Config
aws sagemaker create-endpoint-config \
  --endpoint-config-name car-damage-final-infer-endpoint-config \
  --production-variants VariantName=AllTraffic,ModelName=car-damage-final-infer-model,InitialInstanceCount=1,InstanceType=ml.m5.large \
  --region ap-south-1

# Step 3: Create Endpoint
aws sagemaker create-endpoint \
  --endpoint-name car-damage-final-infer-endpoint \
  --endpoint-config-name car-damage-final-infer-endpoint-config \
  --region ap-south-1
```

---

## Run Inference

python predict.py

NOTE: Make sure your test image is saved as test_image.jpg in the project root.

## Output

🔮 Prediction: 5


## Notes

Ensure IAM role has appropriate SageMaker and S3 permissions.

model.tar.gz must contain both model.pth and inference.py at the root level.

Supported content type: application/x-image.



## 📊 Dashboard (Streamlit UI - optional)

> Coming soon: Upload an image and view prediction using a Streamlit app.

---

## License

MIT License

---

## 🙏 Acknowledgements

* [Roboflow](https://roboflow.com/) for annotation
* [AWS SageMaker](https://aws.amazon.com/sagemaker/) for deployment
* [TorchVision](https://pytorch.org/vision/stable/index.html) for image preprocessing

---

## 🔐 License

MIT

---

Feel free to fork and improve this project. Contributions welcome!

