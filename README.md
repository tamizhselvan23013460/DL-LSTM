# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.


## DESIGN STEPS
### STEP 1: 

Import essential libraries: Load PyTorch, NumPy, Pandas, Matplotlib, and Scikit-learn.

### STEP 2: 

Set device configuration: Choose GPU if available, otherwise fallback to CPU.

### STEP 3: 

Load dataset: Read the CSV file, handle missing values using forward fill.

### STEP 4: 

Extract vocabulary and tags: Collect unique words and entity tags from dataset.

### STEP 5: 

Create index mappings: Build word-to-index, tag-to-index, and reverse tag mappings.

### STEP 6: 

Group words by sentences: Use SentenceGetter to organize words with respective tags.

### STEP 7: 

Convert words and tags to indices: Encode sentences and labels into numerical form.

### STEP 8: 

Pad sequences: Apply padding to ensure uniform sequence lengths across sentences.

### STEP 9: 

Split dataset: Divide data into training and testing sets with defined ratio.

### STEP 10: 

Prepare DataLoader: Define custom Dataset class and create DataLoader objects.

### STEP 11: 

Define BiLSTM model: Create embedding, dropout, bidirectional LSTM, and linear layer.

### STEP 12: 

Train model: Optimize parameters using Adam optimizer and CrossEntropy loss function.

### STEP 13: 

Evaluate and predict: Compare true and predicted tags, then display word-level results.

## PROGRAM

### Name:

### Register Number:

```python
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare data
data = pd.read_csv("ner_dataset.csv", encoding="latin1").ffill()
words = list(data["Word"].unique())
tags = list(data["Tag"].unique())

if "ENDPAD" not in words:
    words.append("ENDPAD")

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
print("Unique tags are:", tags)

# Group words by sentences
class SentenceGetter:
    def _init_(self, data):
        self.grouped = data.groupby("Sentence #", group_keys=False).apply(
            lambda s: [(w, t) for w, t in zip(s["Word"], s["Tag"])]
        )
        self.sentences = list(self.grouped)

getter = SentenceGetter(data)
sentences = getter.sentences

sentences[35]

# Encode sentences
X = [[word2idx[w] for w, t in s] for s in sentences]
y = [[tag2idx[t] for w, t in s] for s in sentences]

word2idx

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

# Pad sequences
max_len = 50
X_pad = pad_sequence([torch.tensor(seq) for seq in X], batch_first=True, padding_value=word2idx["ENDPAD"])
y_pad = pad_sequence([torch.tensor(seq) for seq in y], batch_first=True, padding_value=tag2idx["O"])
X_pad = X_pad[:, :max_len]
y_pad = y_pad[:, :max_len]

X_pad[0]
y_pad[0]
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_pad, test_size=0.2, random_state=1)

# Dataset class
class NERDataset(Dataset):
    def _init_(self, X, y):
        self.X = X
        self.y = y

    def _len_(self):
        return len(self.X)

    def _getitem_(self, idx):
        return {
            "input_ids": self.X[idx],
            "labels": self.y[idx]
        }

train_loader = DataLoader(NERDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(NERDataset(X_test, y_test), batch_size=32)

# Model definition
class BiLSTMTagger(nn.Module):
    def _init_(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self)._init_()
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.dropout=nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, tagset_size)
    def forward(self,x):
        x=self.embedding(x)
        x=self.dropout(x)
        x,_=self.lstm(x)
        return self.fc(x)

model = BiLSTMTagger(len(word2idx)+1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss=0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        train_losses.append(total_loss)
        model.eval()
        val_loss=0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss+=loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss}, Val Loss: {val_loss}")
    return train_losses, val_losses

def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != tag2idx["O"]:
                        true_tags.append(idx2tag[labels[i][j].item()])
                        pred_tags.append(idx2tag[preds[i][j].item()])


# Run training and evaluation
train_losses, val_losses = train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3)
evaluate_model(model, test_loader, X_test, y_test)

# Plot loss
print('Name: EASWAR R')
print('Register Number: 212223230053')
history_df = pd.DataFrame({"loss": train_losses, "val_loss": val_losses})
history_df.plot(title="Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Inference and prediction
i = 125
model.eval()
sample = X_test[i].unsqueeze(0).to(device)
output = model(sample)
preds = torch.argmax(output, dim=-1).squeeze().cpu().numpy()
true = y_test[i].numpy()

print('Name: EASWAR R')
print('Register Number: 212223230053')
print("{:<15} {:<10} {}\n{}".format("Word", "True", "Pred", "-" * 40))
for w_id, true_tag, pred_tag in zip(X_test[i], y_test[i], preds):
    if w_id.item() != word2idx["ENDPAD"]:
        word = words[w_id.item() - 1]
        true_label = tags[true_tag.item()]
        pred_label = tags[pred_tag]
        print(f"{word:<15} {true_label:<10} {pred_label}")

```

### OUTPUT

## Loss Vs Epoch Plot

<img width="756" height="622" alt="image" src="https://github.com/user-attachments/assets/3c672a41-eab7-44e7-bcbc-22572ab3a6ab" />

### Sample Text Prediction:

<img width="441" height="556" alt="image" src="https://github.com/user-attachments/assets/3686412e-e96f-41f6-bb36-1229f009ec7f" />

## RESULT
Thus , an LSTM-based model for recognizing the named entities in the text was excuted Successfully.
