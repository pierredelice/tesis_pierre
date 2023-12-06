"""
Author: Pierre Delice
"""
import re, string, nltk,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import sklearn.model_selection as sk
import sklearn.metrics as skm

# Text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, \
    SpatialDropout1D, Bidirectional
from string import digits

os.environ["mps"] = str("0")


# -----------------------------------------------------------
if not os.path.exists('lstm_results/'):
   output_dir= os.makedirs('lstm_results/')
pd.set_option('display.max_rows', 500)
# -----------------------------------------------------------


# Replace 'your_dataframe.csv' with your actual DataFrame file path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Prepare your Pandas DataFrame

# Load your data
data = pd.read_csv('data/inflacion_result.csv')[['content', 'sent']]

"""# Split your data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = self.dataframe.iloc[idx]['content']  # Adjust column name
        label = self.dataframe.iloc[idx]['sent']  # Adjust column name
        sample = {'features': features, 'labels': label}
        return sample

# Create instances of the custom dataset for training and validation
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)


# Define DataLoader for training and validation
batch_size = 20  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Example of how to iterate through the DataLoader
for batch in train_loader:
    print(batch)
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Step 1: Preprocess the text data
max_sequence_length = 16  # Adjust as needed
max_words = 10000  # Adjust as needed

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['content'])  # Replace 'text_column' with your feature column
sequences = tokenizer.texts_to_sequences(data['content'])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Step 2: Prepare the labels (use label encoding)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['sent'])  # Replace 'label_column' with your label column

# Step 3: Create a custom PyTorch dataset
class CustomTextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        text_sequence = self.sequences[idx]
        labels = self.labels[idx]
        return {'text_sequence': text_sequence, 'labels': labels}

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

train_dataset = CustomTextDataset(X_train, y_train)
val_dataset = CustomTextDataset(X_val, y_val)


# Define DataLoader for training and validation
batch_size = 16  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
len(val_loader.dataset.sequences), len(val_loader.dataset.labels)

# Step 4: Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output[-1, :, :])  # Use the last LSTM output
        return output

# Define model parameters
vocab_size = max_words
embedding_dim = 100  # Adjust as needed
hidden_dim = 16  # Adjust as needed
output_dim = 3  # Three categories

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Step 5: Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30  # Adjust as needed

for epoch in range(num_epochs):
    #model.train()
    total_loss = 0.0

    for batch in train_loader:
        text_sequences = batch['text_sequence']
        labels = batch['labels']
        optimizer.zero_grad()

        outputs = model(text_sequences)
        if len(outputs) == len(labels):
            loss = criterion(outputs, labels)#.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}')

# Evaluation (on validation set)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        text_sequences = batch['text_sequence']
        labels = batch['labels']
        outputs = model(text_sequences)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

#----------------------------------------------------------------

data = pd.read_csv('data/inflacion_result.csv')[['content', 'sent']]

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 10000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['content'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(data['content'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


Y = pd.get_dummies(data['sent']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 20
batch_size = 32

history = model.fit(X_train, Y_train, epochs=epochs, 
                    batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show();


new_complaint = ['el crecimiento economico puede atenuar la inflacion']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['0','1','2']
print(pred, labels[np.argmax(pred)])