import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

def preprocess_img(path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)

model = torch.load('facenet_model_trained.pth', map_location='cpu', weights_only=False)
model.eval()

img_tensor = preprocess_img('IMG_0535.JPG')

with torch.no_grad():
    embedding = model(img_tensor)

print("", embedding.shape)
