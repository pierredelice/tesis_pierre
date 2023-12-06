import pandas as pd
import numpy as np
import re, os, torch
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import sklearn.model_selection as sk
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, hidden_dim, num_classes, lstm_layers, bidirectional, dropout, pad_index, batch_size):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=lstm_layers, bidirectional=bidirectional, batch_first=True)
        num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = num_directions
        self.lstm_units = lstm_units

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)),
                Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)))
        return h, c

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        out = output_unpacked[:, -1, :]
        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds

# Example usage:
# Replace these values with your actual hyperparameters and input dimensions
vocab_size = 10000
embedding_dim = 100
lstm_units = 128
hidden_dim = 64
num_classes = 3
lstm_layers = 2
bidirectional = True
dropout = 0.5
pad_index = 0
batch_size = 32

model = LSTM(vocab_size, embedding_dim, lstm_units, hidden_dim, num_classes, lstm_layers, bidirectional, dropout, pad_index, batch_size)
print(model)


data = pd.read_csv('data/inflacion_result.csv')



# Preprocess your text data and convert it to tensors (example)
text_data = data['content'].tolist()
target_labels = data['sent'].tolist()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Tokenization and Padding (example)
tokenizer = Tokenizer(num_words=vocab_size, 
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(text_data)
X = tokenizer.texts_to_sequences(text_data)
X = pad_sequences(X, maxlen=300)

# Convert target labels to tensors (example)
Y = pd.get_dummies(target_labels).values

# Split your data into train and validation sets
X_train, X_val, Y_train, Y_val = sk.train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert your data to PyTorch tensors and datasets
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

# Create DataLoader for training and validation data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

import nltk 
from nltk.tokenize import word_tokenize

# Tokenize each sentence and calculate their lengths
text_lengths = [len(nltk.word_tokenize(sentence)) for sentence in text_data]

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()

        outputs = model(inputs, batch)  # You may need to compute text_lengths from inputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}')

# Evaluation on validation data
model.eval()
val_loss = 0.0
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch

        outputs = model(inputs, text_lengths)  # You may need to compute text_lengths from inputs
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += len(labels)

validation_accuracy = correct_predictions / total_samples
average_val_loss = val_loss / len(val_loader)

print(f'Validation Loss: {average_val_loss:.4f}')
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')
