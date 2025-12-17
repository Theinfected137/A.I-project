from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import torch
import torch.nn as nn

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("emnist_model.pt", map_location=device))
model.eval()

classes = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(ord('A'), ord('Z')+1)] +
    [chr(i) for i in range(ord('a'), ord('z')+1)]
)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
    transforms.Normalize((0.5,), (0.5,))
])

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    img = Image.open(request.files["image"]).convert("L")
    img = ImageOps.invert(img)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).argmax(dim=1).item()

    return jsonify({"class": classes[pred]})

if __name__ == "__main__":
    app.run(debug=True)
