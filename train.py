import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceDataset(Dataset):
    def __init__(self, folder='facenet_data'):
        self.folder = folder
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.class_to_imgs = {}
        self.data = []

        for person in os.listdir(folder):
            person_path = os.path.join(folder, person)
            if os.path.isdir(person_path):
                images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
                if len(images) >= 2:
                    self.class_to_imgs[person] = images
                    self.data.extend([(img, person) for img in images])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_path, anchor_class = self.data[idx]
        anchor_img = self.transform(Image.open(anchor_path).convert('RGB'))

        positive_path = random.choice([img for img in self.class_to_imgs[anchor_class] if img != anchor_path])
        positive_img = self.transform(Image.open(positive_path).convert('RGB'))

        negative_class = random.choice([cls for cls in self.class_to_imgs if cls != anchor_class])
        negative_path = random.choice(self.class_to_imgs[negative_class])
        negative_img = self.transform(Image.open(negative_path).convert('RGB'))

        return anchor_img, positive_img, negative_img

def train():
    dataset = FaceDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
    model.train()

    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        for anchor, positive, negative in dataloader:
            if anchor.size(0) < 2:
                continue

            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    torch.save(model, 'facenet_model_trained.pth')

if __name__ == "__main__":
    train()
