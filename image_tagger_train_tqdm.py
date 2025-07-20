import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
#from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
#from torchvision.models import resnet50, ResNet50_Weights
#from torchvision.models import resnet34, ResNet34_Weights
#from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.autograd.profiler as profiler
import time
from tqdm import tqdm
import warnings
Image.MAX_IMAGE_PIXELS = 300000000
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")
class JsonlTagDataset(Dataset):
    """
    Dataset for loading image and tag data from a .jsonl file.
    Each line of the file should be a JSON object with keys:
      - "image_path": path to the image file
      - "tags": list of string tags
    """
    def __init__(self, jsonl_file, tag_to_idx, transform=None):
        self.entries = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                obj = json.loads(line)
                self.entries.append(obj)
        self.tag_to_idx = tag_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry['filename']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        tags = entry['tags']
        # Multi-hot encode tags
        y = torch.zeros(len(self.tag_to_idx), dtype=torch.float32)
        for t in tags:
            if t in self.tag_to_idx:
                y[self.tag_to_idx[t]] = 1.0
        return image, y

class PreprocessedTensorDataset(Dataset):
    def __init__(self, jsonl_file, tag_to_idx):
        with open(jsonl_file, 'r') as f:
            self.entries = [json.loads(line) for line in f]
        self.tag_to_idx = tag_to_idx
        self.num_tags = len(tag_to_idx)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        tensor = torch.load(entry['filename'])  # Already preprocessed [C, H, W]
        tags = entry['tags']

        target = torch.zeros(self.num_tags, dtype=torch.float32)
        for tag in tags:
            if tag in self.tag_to_idx:
                target[self.tag_to_idx[tag]] = 1.0

        return tensor, target
# Build tag vocabulary

def build_tag_vocab(jsonl_file, min_freq=1):
    freq = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            for t in obj['tags']:
                freq[t] = freq.get(t, 0) + 1
    # Filter by freq
    tags = [t for t, c in freq.items() if c >= min_freq]
    tag_to_idx = {t: i for i, t in enumerate(sorted(tags))}
    return tag_to_idx

# Training function with inner progress bar

# ~ def train(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    # ~ model.train()
    # ~ running_loss = 0.0
    # ~ # progress over batches
    # ~ for images, targets in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=False):
        # ~ images = images.to(device)
        # ~ targets = targets.to(device)
        # ~ optimizer.zero_grad()
        # ~ outputs = model(images)
        # ~ loss = criterion(outputs, targets)
        # ~ loss.backward()
        # ~ optimizer.step()
        # ~ running_loss += loss.item() * images.size(0)
    # ~ epoch_loss = running_loss / len(dataloader.dataset)
    # ~ return epoch_loss

def train(model, dataloader, criterion, optimizer, device, epoch, num_epochs, profile=False):
    model.train()
    running_loss = 0.0
    scaler = GradScaler()

    for i, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=False)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        # ~ # Profile only first batch of first epoch
        # ~ if profile and epoch == 1 and i == 0:
            # ~ print("Profiling first batch...")
            # ~ with profiler.profile(use_device='cuda') as prof:
                # ~ with autocast('cuda'):
                    # ~ outputs = model(images)
                    # ~ loss = criterion(outputs, targets)
                # ~ scaler.scale(loss).backward()
                # ~ scaler.step(optimizer)
                # ~ scaler.update()
            # ~ prof.export_chrome_trace("profile_trace.json")
            # ~ print("Profiler trace saved to profile_trace.json")
            # ~ running_loss += loss.item() * images.size(0)
            # ~ continue  # skip rest of loop for this batch
        # ~ else:
            # ~ with autocast('cuda'):
                # ~ outputs = model(images)
                # ~ loss = criterion(outputs, targets)
            # ~ scaler.scale(loss).backward()
            # ~ scaler.step(optimizer)
            # ~ scaler.update()
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Validation function

def log_per_tag_f1(outputs, targets, idx_to_tag, file_path=None):
    """
    Logs per-tag F1 scores, sorted by F1 descending. Optionally writes to a file.

    Args:
        outputs (Tensor): Raw model outputs (after sigmoid)
        targets (Tensor): Ground truth labels (multi-hot)
        idx_to_tag (Dict[int, str]): Mapping from index to tag name
        file_path (str, optional): If set, writes results to this file
    """
    pred = (outputs >= 0.5).float()
    tp = (pred * targets).sum(dim=0)
    fp = (pred * (1 - targets)).sum(dim=0)
    fn = ((1 - pred) * targets).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Collect tag, F1 as tuples
    tag_f1_pairs = [(idx_to_tag[i], f1[i].item()) for i in range(len(f1))]
    tag_f1_pairs.sort(key=lambda x: x[1], reverse=True)

    lines = [f"{tag}: F1 = {score:.4f}" for tag, score in tag_f1_pairs]
    output_text = "\n".join(lines)

    if file_path:
        with open(file_path, 'a') as f:
            f.write(output_text + "\n\n")
    else:
        print(output_text)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)
            all_outputs.append(torch.sigmoid(logits).cpu())  # store probs
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    outputs = torch.cat(all_outputs, dim=0)  # [N, T]
    targets = torch.cat(all_targets, dim=0)  # [N, T]

    # compute mean F1 as before (optional)
    pred = (outputs >= 0.5).float()
    tp = (pred * targets).sum(dim=0)
    fp = (pred * (1 - targets)).sum(dim=0)
    fn = ((1 - pred) * targets).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    mean_f1   = f1.mean().item()

    return epoch_loss, mean_f1, outputs, targets

class SmoothBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.01):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # Smooth the 0/1 targets towards 0.5
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.BCEWithLogitsLoss()(pred, target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()

class TaggingNetRes(nn.Module):
    def __init__(self, feature_extractor: nn.Module, num_tags: int):
        super().__init__()
        self.features = feature_extractor
        self.dropout = nn.Dropout(p=0.3)
        # The feature_extractor outputs [B, 512, 1, 1], so flatten to [B, 512].
        feat_dim = 512  # for ResNet-18, this is always 512
        self.classifier = nn.Linear(feat_dim, num_tags)

    def forward(self, x):
        # x: [B, 3, H, W] → features(x): [B, 512, 1, 1]
        x = self.features(x)             # → [B, 512, 1, 1]
        x = x.view(x.size(0), -1)        # → [B, 512]
        x = self.dropout(x)              # → [B, 512]
        return self.classifier(x)

class TaggingNetEff(nn.Module):
    def __init__(self, feature_extractor: nn.Module, feat_dim: int, num_tags: int):
        super().__init__()
        self.features = feature_extractor            # EfficientNet’s conv → pool
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(feat_dim, num_tags)

    def forward(self, x):
        # x: [B, 3, H, W]  →  features(x): [B, 1280, 1, 1]
        x = self.features(x).view(x.size(0), -1)      # → [B, 1280]
        x = self.dropout(x)                           # → [B, 1280]
        return self.classifier(x)                     # → [B, num_tags]

if __name__ == '__main__':
    # Config
    jsonl_file = '/home/simon/misc_apps/Linux/AI_TEST/train/tags.jsonl'
    batch_size = 16
    num_epochs = 20
    lr = 1e-4
    imgsize = 300
    name = "anime_effnetB3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    #transform = transforms.Compose([transforms.Resize((imgsize, imgsize)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #use with resnet-X
    #transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    transform = EfficientNet_B3_Weights.DEFAULT.transforms() #use with efficientnet-bX
    tag_to_idx = build_tag_vocab(jsonl_file, min_freq=50)
    idx_to_tag = {i: t for t, i in tag_to_idx.items()}
    dataset = JsonlTagDataset(jsonl_file, tag_to_idx, transform)
    # Split dataset (80/20)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=6)

    # Model: pre-trained ResNet or EfficientNet
    #base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    base_model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    #base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    #base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    #base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_tags = len(tag_to_idx)

    feature_extractor = nn.Sequential(base_model.features, base_model.avgpool) #use for efficientnet_bX
    feat_dim = base_model.classifier[1].in_features  # should be 1280 for B0
    model = TaggingNetEff(feature_extractor, feat_dim, num_tags).to(device)

    #base_model.fc = nn.Linear(base_model.fc.in_features, num_tags) #use for resnet-X
    #feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
    #model = TaggingNet(feature_extractor, num_tags).to(device)
    #model = base_model.to(device)

    #criterion = nn.BCEWithLogitsLoss()
    #criterion = SmoothBCEWithLogitsLoss(smoothing=0.01)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3)

    # Training loop (no outer progress bar)
    best_f1 = 0.0
    start_time = time.time()
    # ~ for epoch in range(1, num_epochs + 1):
        # ~ epoch_start = time.time()
        # ~ train_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        # ~ val_loss, val_f1, outputs, targets = validate(model, val_loader, criterion, device)
        # ~ old_lr = optimizer.param_groups[0]['lr']
        # ~ scheduler.step(val_loss)
        # ~ new_lr = optimizer.param_groups[0]['lr']
        # ~ if new_lr != old_lr:
            # ~ print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        # ~ #scheduler.step()
        # ~ epoch_time = time.time() - epoch_start
        # ~ remaining_epochs = num_epochs - epoch
        # ~ est_remaining = remaining_epochs * epoch_time
        # ~ est_h = int(est_remaining // 3600)
        # ~ est_m = int((est_remaining % 3600) // 60)
        # ~ est_s = int(est_remaining % 60)
        # ~ print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean Val-F1: {val_f1:.4f}")
        # ~ print(f"Estimated time remaining: {est_h:02d}:{est_m:02d}:{est_s:02d}")
        # ~ log_per_tag_f1(outputs, targets, idx_to_tag, f"./{num_tags}tag_log_epoch_{epoch:03d}_meanF1_{val_f1:.4f}")
        # ~ # Save best model
        # ~ if val_f1 > best_f1:
            # ~ best_f1 = val_f1
            # ~ torch.save(model.state_dict(), f'{num_tags}tags_{lr}lr_{imgsize}x_{name}.pth')

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, profile=(epoch == 1))
        val_loss, val_f1, outputs, targets = validate(model, val_loader, criterion, device)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        epoch_time = time.time() - epoch_start
        remaining_epochs = num_epochs - epoch
        est_remaining = remaining_epochs * epoch_time
        est_h = int(est_remaining // 3600)
        est_m = int((est_remaining % 3600) // 60)
        est_s = int(est_remaining % 60)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean Val-F1: {val_f1:.4f}")
        print(f"Estimated time remaining: {est_h:02d}:{est_m:02d}:{est_s:02d}")
        log_per_tag_f1(outputs, targets, idx_to_tag, f"./{num_tags}tag_log_epoch_{epoch:03d}_meanF1_{val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'{num_tags}tags_{lr}lr_{imgsize}x_{name}.pth')

    total_time = time.time() - start_time
    total_h = int(total_time // 3600)
    total_m = int((total_time % 3600) // 60)
    total_s = int(total_time % 60)
    print("Training complete. Best Val F1: {:.4f}".format(best_f1))
    print("Total training time: {:02d}:{:02d}:{:02d}".format(total_h, total_m, total_s))

    # Save tag vocabulary for inference
    with open(f'tag_to_idx_{num_tags}tags_{name}.json', 'w') as f:
        json.dump(tag_to_idx, f)
