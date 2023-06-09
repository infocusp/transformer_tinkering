# Import necessary libraries
# from google.colab import drive
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

# Mount the google drive to access files
# drive.mount('/content/drive')

# Load the dataset
datapath = f'bbc-text.csv'
df = pd.read_csv(datapath)

# Visualize the distribution of categories
# df.groupby(['category']).size().plot.bar()

# Instantiate the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Define a mapping of categories to numerical labels
labels = {'business':0, 'entertainment':1, 'sport':2, 'tech':3, 'politics':4}

# Define a PyTorch Dataset class for our specific dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        # Tokenize the texts and store the labels
        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['text']]

    def __len__(self):
        # Return the size of the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Fetch an example by its index
        return self.texts[idx], self.labels[idx]

# Define a BERT-based classifier
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        # Define the BERT model, a dropout layer, and a final linear layer for classification
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)

    def forward(self, input_id, mask):
        # Pass the inputs through the BERT model
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)

        # Apply the dropout and linear layers to the output
        dropout_output = self.dropout(pooled_output)
        final_output = self.linear(dropout_output)

        return final_output

# Function to train the model
def train(model, train_data, val_data, learning_rate, epochs):
    # Prepare the data, dataloaders and the model
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr= learning_rate)

    # Training loop
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        # Training step
        for train_input, train_label in tqdm(train_dataloader):
            # Move the data to GPU
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            # Perform a forward pass through the model
            output = model(input_id, mask)

            # Compute the loss and the accuracy
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            # Backward step
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # Validation step
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                # Move the data to GPU
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                # Perform a forward pass through the model
                output = model(input_id, mask)

                # Compute the loss and the accuracy
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        # Print statistics for this epoch
        print(f'Epoch: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data):.3f} | Train Accuracy: {total_acc_train / len(train_data):.3f} | Val Loss: {total_loss_val / len(val_data):.3f} | Val Accuracy: {total_acc_val / len(val_data):.3f}')

# Function to evaluate the model
def evaluate(model, test_data):
    # Prepare the data and the model
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Evaluation loop
    total_acc_test = 0

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            # Move the data to GPU
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            # Perform a forward pass through the model
            output = model(input_id, mask)

            # Compute the accuracy
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    # Print the test accuracy
    print(f'Test Accuracy: {total_acc_test / len(test_data):.3f}')

# Split the dataframe into training, validation and test sets
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])

# Instantiate the model
model = BertClassifier()

# Train the model
train(model, df_train, df_val, 1e-6, 5)

# Evaluate the model
evaluate(model, df_test)
