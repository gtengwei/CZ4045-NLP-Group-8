import gensim.downloader
from gensim.models import Word2Vec
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

import tqdm
import time
from datetime import datetime
import pickle
import os
from itertools import repeat
from torch.utils.tensorboard import SummaryWriter

from gensim.models import KeyedVectors
# Download 'word2vec-google-news-300' embeddings, or load it if it's there alr

VECTOR_PATH = "word2vec-google-news-300.kv"
if os.path.exists(VECTOR_PATH):
    google_news_vectors = KeyedVectors.load(VECTOR_PATH)
else:
    google_news_vectors = gensim.downloader.load('word2vec-google-news-300')
    google_news_vectors.save(VECTOR_PATH)

# Load all the data required
with open("BIO_train.txt", "r") as f:
    data_train = f.readlines()
with open("BIO_development.txt", "r") as f:
    data_dev = f.readlines()
with open("BIO_test.txt", "r") as f:
    data_test = f.readlines()
# Find the maximum sentence length to determine LSTM input size
max_length = 0
count = 0
tag_set = set()
for i in data_train + data_dev + data_test:
    if i != "\n":
        count += 1
        tag_set.add(i.split()[-1])
    else:
        max_length = max(max_length, count)
        count = 0
        
# Find the embedding dimension
vec = google_news_vectors["Test"]

print(f"Maximum Sentence Length: {max_length}, Embedding Shape: {vec.shape}, No. Labels: {len(tag_set)}")

# Build simple LSTM model
# Define constants/params
HIDDEN_SIZE = 32
NUM_LAYERS = 1
OUTPUT_SIZE = len(tag_set)
MAX_LENGTH = 150 # Max sequence length in dataset is 124
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training related
BATCH_SIZE = 1024
# Simple LSTM model
class simple_lstm(nn.Module):
    def __init__(
                    self, 
                    input_size= 300, 
                    hidden_size= HIDDEN_SIZE, 
                    output_size= OUTPUT_SIZE,
                    num_layers= NUM_LAYERS,
                    # dropout_rate= 0.2, 
                    bidirectional= True, 
                ):
        
        super().__init__()
        # Model body
        self.lstm = nn.LSTM(input_size= input_size, hidden_size= hidden_size, bidirectional= bidirectional, num_layers= num_layers)
        
        # Model head
        self.head = nn.Sequential(
            nn.Linear((2 if bidirectional else 1) * hidden_size, output_size),
            # nn.Dropout(dropout_rate),
            nn.Softmax(dim= 2)
        )
    
    def __call__(self, input):
        return self.head(self.lstm(input)[0])

# One-hot encode labels
# Mapping from tagset to embedding
label_map = {label:idx for idx, label in enumerate(sorted(list(tag_set)))}
index_map = {idx:label for idx, label in enumerate(sorted(list(tag_set)))}
encoder = OneHotEncoder(sparse_output= False)
encoder.fit(np.array(list(range(9))).reshape(-1, 1))

# Create Dataset
class NERDataset(Dataset):
    # TODO: Try out using packed sequences to reduce computation (but now seems like loss and score cal is bottleneck so there's that)
    def __init__(self, file, embedding= google_news_vectors, label_map= label_map, max_length= MAX_LENGTH):
        # Split into X and y
        self.X = [] # Shape = (Sentence, word, len(embedding)= 300)
        self.y = [] # Shape = (Sentence, Word)
        self.max_length = max_length
        sentence_words = []
        sentence_tags = []
        self.pad_width = []
        for line in tqdm.tqdm(file):
            if line != "\n":
                word, tag = line.split()
                # Get the embeddings for each word (X)
                try:
                    sentence_words.append(embedding[word])
                except KeyError:
                    sentence_words.append(embedding["UNK"])
                # Convert the tag into index (y)
                sentence_tags.append([label_map[tag]])
            else:
                # Pad to ensure that the sequence length is the same for easy batching
                # The dim that is padded corresponds to the sequence length aka number of words
                # Front padding is used as it performs better
                pad_width = self.max_length - len(sentence_words)
                self.X.append(np.pad(np.array(sentence_words), ((pad_width, 0), (0, 0)), mode= "constant", constant_values= [0]))
                # self.y.append(np.pad(encoder.transform(np.array(sentence_tags)), ((pad_width, 0), (0, 0)), mode= "constant", constant_values= [0]))
                self.y.append(np.pad(np.array(sentence_tags, dtype= np.longlong), ((pad_width, 0), (0, 0)), mode= "constant", constant_values= [-100]))
                self.pad_width.append(pad_width)
                # self.X.append(np.array(sentence_words))
                # self.y.append(encoder.transform(np.array(sentence_tags)))
                sentence_words = []
                sentence_tags = []
        # If dataset does not end with \n
        if sentence_words:
            pad_width = self.max_length - len(sentence_words)
            self.X.append(np.pad(np.array(sentence_words), ((pad_width, 0), (0, 0)), mode= "constant", constant_values= [0]))
            self.y.append(np.pad(np.array(sentence_tags, dtype= np.longlong), ((pad_width, 0), (0, 0)), mode= "constant", constant_values= [-100]))
            self.pad_width.append(pad_width)
            sentence_words = []
            sentence_tags = []
        
        # Convert to tensors
        self.X = torch.tensor(np.array(self.X), device= DEVICE)
        self.y = torch.tensor(np.array(self.y), device= DEVICE).squeeze(2)
        self.pad_width = torch.tensor(np.array(self.pad_width), device= DEVICE)
        
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.pad_width[idx] 

# Create data folder if not exist
if not os.path.exists("./data/"):
    os.makedirs("./data/")

# Load if already there else reinstantiate
# Train
if os.path.exists("./data/train_ds.pkl"):
    with open("./data/train_ds.pkl", "rb") as f:
        train_ds = pickle.load(f)
else:
    train_ds = NERDataset(data_train)
    with open("./data/train_ds.pkl", "wb") as f:
        pickle.dump(train_ds, f)
# Dev
if os.path.exists("./data/dev_ds.pkl"):
    with open("./data/dev_ds.pkl", "rb") as f:
        dev_ds = pickle.load(f)
else:
    dev_ds = NERDataset(data_dev)
    with open("./data/dev_ds.pkl", "wb") as f:
        pickle.dump(dev_ds, f)
# Test
if os.path.exists("./data/test_ds.pkl"):
    with open("./data/test_ds.pkl", "rb") as f:
        test_ds = pickle.load(f)
else:
    test_ds = NERDataset(data_test)
    with open("./data/test_ds.pkl", "wb") as f:
        pickle.dump(test_ds, f)

# Create Dataloaders
train_dataloader = DataLoader(train_ds, BATCH_SIZE, shuffle= True)
dev_dataloader = DataLoader(dev_ds, BATCH_SIZE, shuffle= True)
test_dataloader = DataLoader(test_ds, BATCH_SIZE, shuffle= True)

# Training

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
            
def train_model(model, train_dataloader, val_dataloader, early_stop= True, n_epochs= 100):
    run_time = datetime.now().strftime('%d_%m_%y_%H%M%S')
    writer = SummaryWriter(f"runs/{run_time}")
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    # In case anything goes wrong
    torch.autograd.set_detect_anomaly(True)

    # Implement early stopping
    stopper = EarlyStopper()

    # Keep track of epoch loss/acc
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    epoch_time = []
    
    batch_count = 0
    
    for epoch in tqdm.tqdm(range(n_epochs)):
        tic = time.time()
        # Keep track of batch loss/acc
        train_loss_epoch = []
        train_acc_epoch = []
        test_loss_epoch = []
        test_acc_epoch = []
        model.train()
        for batch in train_dataloader:
            # print("In batch")
            # take a batch
            X_batch, y_batch, pad_batch = batch
            toc2 = time.time()
            # forward pass
            y_pred = model(X_batch)
            toc = time.time()
            # print(f"Time to forward pass: {toc-toc2}")
            # print("Passed")
            # loss = nn.functional.cross_entropy(y_pred, y_batch, ignore_index= -100)
            loss = nn.functional.cross_entropy(torch.swapaxes(y_pred, 1, 2), y_batch, ignore_index= -100)
            train_loss_epoch.append(loss.detach().cpu())
            toc2 = time.time()
            # print(f"Time to calculate loss: {toc2 - toc}")
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            toc = time.time()
            # print(f"Time to calculate backward: {toc - toc2}")
            
            # update weights
            optimizer.step()
            toc2 = time.time()
            # print(f"Time to step: {toc2 - toc}")
            
            # Decode the sentences and get the f1 score using seqeval
            # Get the labels, shape = (sentence, word, categories) -> (sentence, word), where each word is a label from 0 - 9
            target_idx = y_batch
            # target_idx = torch.max(y_batch, 2)[1]
            pred_idx = torch.max(y_pred, 2)[1]
            
            vectorized_fn = np.vectorize(index_map.get)
            # Map the np arrays into np arrays of strings/labels, then unpad them
            f1 = f1_score([sentence[pad_batch[idx]:].tolist() for idx, sentence in enumerate(vectorized_fn(target_idx.detach().cpu().int()))], 
                        [sentence[pad_batch[idx]:].tolist() for idx, sentence in enumerate(vectorized_fn(pred_idx.detach().cpu().int()))])
            
            toc = time.time()
            # print(f"Time to score: {toc - toc2}")
            train_acc_epoch.append(f1)
            
            writer.add_scalar("Train Loss", loss.detach(), batch_count)
            batch_count += 1
            
        # Calculate the epoch acc and loss
        train_loss.append(np.mean(train_loss_epoch))
        train_acc.append(np.mean(train_acc_epoch))
        
        
        # Calculate for test set as well
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                # take a batch
                X_batch, y_batch, pad_batch = batch
                # forward pass
                y_pred = model(X_batch)
                # loss = unpad_CrossEntropyLoss(y_pred, y_batch, pad_batch)
                loss = nn.functional.cross_entropy(torch.swapaxes(y_pred, 1, 2), y_batch, ignore_index= -100)

                
                test_loss_epoch.append(loss.detach().cpu())
                
                target_idx = y_batch
                # target_idx = torch.max(y_batch, 2)[1]
                pred_idx = torch.max(y_pred, 2)[1]
                
                vectorized_fn = np.vectorize(index_map.get)
                # Map the np arrays into np arrays of strings/labels, then unpad them
                f1 = f1_score([sentence[pad_batch[idx]:].tolist() for idx, sentence in enumerate(vectorized_fn(target_idx.detach().cpu().int()))], 
                            [sentence[pad_batch[idx]:].tolist() for idx, sentence in enumerate(vectorized_fn(pred_idx.detach().cpu().int()))])
                test_acc_epoch.append(f1)
                
        # Calculate the epoch acc and loss
        test_loss.append(np.mean(test_loss_epoch))
        test_acc.append(np.mean(test_acc_epoch))
        # print(f"Epoch: {epoch} Train Loss: {train_loss[-1]} Test Loss: {test_loss[-1]}")
        
        epoch_time.append(time.time() - tic)
        
        # Break loop if early stopping in activate
        if stopper.early_stop(test_loss[-1]) and early_stop:
            print(f"Early stop at epoch {epoch + 1}/{n_epochs}")
            break
        
        writer.add_scalars("Epoch Loss", {"Train Loss":train_loss[-1], "Val Loss":test_loss[-1]}, epoch)
        writer.add_scalars("Epoch F1", {"Train F1":train_acc[-1], "Val F1":test_acc[-1]}, epoch)
        
        if test_acc[-1] == min(test_acc):
            best_model = model.state_dict()
    
    # Return last epoch's acc and time
    return train_loss, train_acc, test_loss, test_acc, epoch_time, best_model

model = simple_lstm().to(DEVICE)
train_loss, train_acc, test_loss, test_acc, epoch_time, best_model = train_model(model, train_dataloader, dev_dataloader)

# Create folder if not exist
if not os.path.exists("./train_results/"):
    os.makedirs("./train_results/")
with open(f"train_results/{datetime.now().strftime('%d_%m_%y_%H%M%S')}.pkl", "wb") as f:
    pickle.dump((train_loss, train_acc, test_loss, test_acc, epoch_time, best_model), f)