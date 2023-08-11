#!/usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")


def main():
    csv_path = os.path.join(os.getcwd(), 'data', 'HIV.csv')
    df = pd.read_csv(csv_path)
    print("Data Loaded")
    del df['activity'] # removing the name of the molecule since that doesn't matter
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
    print("Train Test Split Finished")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_classes)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Model and Tokenizer Initialized")

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
    print("Data tokenization and preprocessing complete")

    # Create DataLoader
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train DataLoader created")

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    label_columns = list(df.columns)
    label_columns.remove("smiles")

    label_accuracies = {label: 0.0 for label in label_columns}
    label_sample_counts = {label: 0 for label in label_columns}

    """
    Cross Entropy Loss instead of BCE
    Calculate Loss for each column inside a loop
    """
    # Training loop

    print("STARTING TRAINING")
    model.train()
    for epoch in range(num_epochs):
        total_samples = 0
        for batch_input_ids, batch_attention_masks, batch_labels in train_loader:
            optimizer.zero_grad()

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            batch_labels = batch_labels.to(device)  

            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            predicted_labels = outputs.logits.argmax(dim=1)
            total_loss = 0.0

            for label_idx, label in enumerate(label_columns):
                correct_labels = (predicted_labels == batch_labels[:, label_idx]).sum().item() #calculates the sum of "true" in the created tensor
                label_accuracies[label] += correct_labels
                label_sample_counts[label] += len(batch_labels)

                column_predictions = outputs.logits[:, label_idx]  # Predictions for this column
                column_labels = batch_labels[:, label_idx]  # True labels for this column
                loss = criterion(column_predictions, column_labels)
                total_loss += loss


            print(f"Epoch {epoch+1}/{num_epochs} - Batch Loss: {total_loss:.4f}")
            loss.backward()
            optimizer.step()
        
        for label in label_columns:
            label_accuracy = label_accuracies[label] / label_sample_counts[label]
            print(f"Accuracy for {label}: {label_accuracy:.4f}")

    # Example SMILES string
    torch.save(model.state_dict(), os.path.join(os.getcwd(), 'models', 'HIV.pth'))

if __name__ == "__main__":
    main()


