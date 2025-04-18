from train import FaceDataset
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('facenet_model_trained.pth', map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2)

def evaluate(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            similarity_pos = cosine_similarity(anchor_emb, positive_emb)
            similarity_neg = cosine_similarity(anchor_emb, negative_emb)

            correct += (similarity_pos > similarity_neg).sum().item()
            total += anchor.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

dataset = FaceDataset(folder='facenet_data')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

evaluate(model, dataloader)
