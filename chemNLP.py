#!/usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def main():
    csv_path = '/Users/siddharthmaddikayala/machine_learning/data/tox21.csv'
    df = pd.read_csv(csv_path)
    del df['mol_id'] # removing the name of the molecule since that doesn't matter
    df.replace({'true': 1, 'false': 0}, inplace=True)
    df.fillna(-1, inplace=True) #changing "Nan" to -1

    num_classes = len(df.columns) - 1
    max_sequence_length = df['smiles'].apply(len).max()
    #print(df.head())
    #print(vocab_size)
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001



    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_classes)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and preprocess data
    encoded_data_train = tokenizer.batch_encode_plus(
        train_df['smiles'].tolist(),
        add_special_tokens=True,
        padding=True,
        return_attention_mask=True,
        max_length=max_sequence_length,
        return_tensors='pt',
        truncation=True
    )
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(train_df.drop(columns=['smiles']).values, dtype=torch.float)

    # Create DataLoader
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_input_ids, batch_attention_masks, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = (outputs.logits > 0.5).float()  # Threshold logits for binary classification
            batch_correct = (predicted_labels == batch_labels).all(dim=1).sum().item()
            total_correct += batch_correct
            total_samples += len(batch_labels)
        
        average_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {average_loss:.4f} - Train Accuracy: {accuracy:.4f}")

    # Example SMILES string
    torch.save(model.state_dict(), "/Users/siddharthmaddikayala/machine_learning/models/tox21.pth")

def test():
    df = pd.read_csv('/Users/siddharthmaddikayala/machine_learning/data/tox21.csv')

    del df['mol_id']
    df.replace({'true': 1, 'false': 0}, inplace=True)
    df.fillna(-1, inplace=True)

    # Define constants
    num_classes = len(df.columns) - 1  # Number of labels excluding the text column
    max_sequence_length = df['smiles'].apply(len).max()
    batch_size = 32 

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize and preprocess test data
    encoded_data_test = tokenizer.batch_encode_plus(
        test_df['smiles'].tolist(),
        add_special_tokens=True,
        padding=True,
        return_attention_mask=True,
        max_length=max_sequence_length,
        return_tensors='pt',
        truncation=True
    )
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(test_df.drop(columns=['smiles']).values, dtype=torch.float)

    # Create DataLoader for testing
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the saved model
    model.load_state_dict(torch.load('/Users/siddharthmaddikayala/machine_learning/models/tox21.pth'))
    model.eval()

    # Test the model
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_input_ids, batch_attention_masks, batch_labels in test_loader:
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            predicted_labels = (outputs.logits > 0.5).float()  # Threshold logits for binary classification
            correct_predictions += (predicted_labels == batch_labels).all(dim=1).sum().item()
            total_samples += len(batch_labels)

    accuracy = correct_predictions / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
   


if __name__ == "__main__":
    main()


