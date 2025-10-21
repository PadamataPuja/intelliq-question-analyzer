import pandas as pd
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast

# CPU optimization
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("Running on CPU - Optimizing for multi-core")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)

train_df = pd.read_csv(r"dataset\train.csv")
valid_df = pd.read_csv(r"dataset\valid.csv")

# Clean HTML tags
def clean_html(text):
    return re.sub('<.*?>', '', str(text))

train_df['text'] = (train_df['Title'] + " " + train_df['Body']).apply(clean_html)
valid_df['text'] = (valid_df['Title'] + " " + valid_df['Body']).apply(clean_html)

# Encode labels
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['Y'])
valid_df['label'] = le.transform(valid_df['Y'])

# Print class-to-label mapping
for i, class_name in enumerate(le.classes_):
    print(f"{class_name} => {i}")

max_len = 128
batch_size = 16  # Increased batch size for better efficiency
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class StackOverflowDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = StackOverflowDataset(train_df['text'].values, train_df['label'].values, tokenizer, max_len)
valid_dataset = StackOverflowDataset(valid_df['text'].values, valid_df['label'].values, tokenizer, max_len)

# Create data loaders (Windows compatible)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training configuration
EPOCHS = 15
MAX_TRAIN_BATCHES = 200
MAX_VAL_BATCHES = 50  # ⭐ LIMIT VALIDATION BATCHES - This is the key optimization!

# Tracking lists
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("Starting training...")
for epoch in range(EPOCHS):
    # ========== TRAINING PHASE ==========
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Use limited training batches
    limited_train_loader = islice(train_loader, MAX_TRAIN_BATCHES)
    
    train_pbar = tqdm(limited_train_loader, total=MAX_TRAIN_BATCHES, 
                     desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
    
    for batch in train_pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    train_loss = total_loss / MAX_TRAIN_BATCHES
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ========== VALIDATION PHASE (OPTIMIZED) ==========
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    # ⭐ KEY OPTIMIZATION: Limit validation batches
    limited_val_loader = islice(valid_loader, MAX_VAL_BATCHES)
    
    val_pbar = tqdm(limited_val_loader, total=MAX_VAL_BATCHES, 
                   desc=f"Epoch {epoch+1}/{EPOCHS} - Validation")

    with torch.no_grad():
        for batch in val_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
            # Update progress bar
            val_pbar.set_postfix({
                'loss': f'{outputs.loss.item():.4f}',
                'acc': f'{val_correct/val_total:.4f}'
            })

    val_loss /= MAX_VAL_BATCHES  # Changed from len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print("-" * 80)

print("Training completed!")

# Plotting accuracy and loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_loss.png", dpi=300)
plt.show()

# ========== FULL EVALUATION ON TEST SET ==========
print("Running full evaluation on test set...")

# Create proper test dataset (you were using train data before)
test_dataset = StackOverflowDataset(valid_df['text'].values, valid_df['label'].values, tokenizer, max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Accuracy
test_accuracy = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
labels_unique = sorted(np.unique(all_labels))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_unique, yticklabels=labels_unique)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("Confusion_matrix.png", dpi=300)
plt.show()

# Save the model
torch.save(model.state_dict(), "bert_stackoverflow_model.pth")