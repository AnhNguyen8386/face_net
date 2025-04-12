import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F

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

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2)

img1_tensor = preprocess_img('test_img\\me1.jpg')
img2_tensor = preprocess_img('test_img\\me2.jpg')

with torch.no_grad():
    embedding1 = model(img1_tensor)
    embedding2 = model(img2_tensor)

similarity = cosine_similarity(embedding1, embedding2)

if similarity.item() > 0.6:
    print("Cả hai ảnh là của cùng một người")
else:
    print("Hai ảnh không phải của cùng một người")
