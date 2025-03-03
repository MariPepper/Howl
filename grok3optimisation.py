import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
import torchaudio
from torch.utils.data import DataLoader, TensorDataset

class TextProcessor(nn.Module):
    def __init__(self):
        super(TextProcessor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, text):
        if isinstance(text, list):
            text = [t for t in text]
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(text.device if hasattr(text, 'device') else 'cpu')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

class ImageProcessor(nn.Module):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Identity()
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, image):
        image = self.transform(image)
        return self.model(image.unsqueeze(0) if image.dim() == 3 else image)

class AudioProcessor(nn.Module):
    def __init__(self):
        super(AudioProcessor, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=80, stride=4)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(4)
        self.fc_input_dim = self._calculate_fc_input_dim(309)
        self.fc = nn.Linear(self.fc_input_dim, 512)
        self.bn = nn.BatchNorm1d(128)

    def _calculate_fc_input_dim(self, audio_len):
        x = torch.randn(1, 1, audio_len)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        return x.view(1, -1).size(1)

    def forward(self, audio):
        x = torch.relu(self.conv1(audio.unsqueeze(1)))
        x = self.bn(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))

class MultimodalFusion(nn.Module):
    def __init__(self, text_dim, image_dim, audio_dim, output_dim):
        super(MultimodalFusion, self).__init__()
        
        self.text_proj = nn.Linear(text_dim, 512)
        self.image_proj = nn.Linear(image_dim, 512)
        self.audio_proj = nn.Linear(audio_dim, 512)
        
        # Attention mechanism for better fusion
        self.attention = nn.MultiheadAttention(512, num_heads=8)
        
        # Final fusion layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, text_features, image_features, audio_features):
        # Project features to a common dimension
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        audio_proj = self.audio_proj(audio_features)
        
        # Stack the features for attention
        combined = torch.stack([text_proj, image_proj, audio_proj], dim=1)
        
        # Apply attention across modalities
        attn_output, _ = self.attention(combined, combined, combined)
        # Use mean to get a single vector from attention output
        fused = attn_output.mean(dim=1)
        
        # Further process the fused features
        fused = self.dropout(torch.relu(self.fc1(fused)))
        return self.fc2(fused)

class Grok3(nn.Module):
    def __init__(self):
        super(Grok3, self).__init__()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.fusion = MultimodalFusion(768, 2048, 512, 512)
        self.classifier = nn.Linear(512, 10)

    def forward(self, text, image, audio):
        text_features = self.text_processor(text)
        image_features = self.image_processor(image)
        audio_features = self.audio_processor(audio)
        fused_features = self.fusion(text_features, image_features, audio_features)
        return self.classifier(fused_features)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Grok3().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Dummy data for illustration
text = ["This is a test sentence.", "Another sentence."] * 32
image = torch.randn(32, 3, 224, 224)
audio = torch.randn(32, 309)
labels = torch.randint(0, 10, (32,))

# Create dataset and dataloader
dataset = TensorDataset(torch.tensor([1 for _ in text]), image, audio, labels)  # Placeholder for text tensor
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop with validation
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        text_batch, image_batch, audio_batch, labels_batch = [x.to(device) for x in batch]
        outputs = model(text, image_batch, audio_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    scheduler.step()
    train_loss /= len(dataloader)

    # Validation - Here you would ideally have a separate validation dataset
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            text_batch, image_batch, audio_batch, labels_batch = [x.to(device) for x in batch]
            val_outputs = model(text, image_batch, audio_batch)
            val_loss += criterion(val_outputs, labels_batch).item()
    val_loss /= len(dataloader)

    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")