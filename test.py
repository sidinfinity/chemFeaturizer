#!/usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from data.HIV_data import load_HIV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

def main():
    labels, df = load_HIV()

    # Define constants
    num_labels = len(df.columns) - 1
    num_classes = 2 # Number of labels excluding the text column
    max_sequence_length = df['smiles'].apply(len).max()
    batch_size = 32 

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.to(device)
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
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models", "HIV.pth")))
    model.eval()

    # Test the model
    label_accuracies = {label: 0.0 for label in labels}
    label_sample_counts = {label: 0 for label in labels}

    with torch.no_grad():
        for batch_input_ids, batch_attention_masks, batch_labels in test_loader:
            
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            predicted_labels = (outputs.logits > 0.5).float()  # Threshold logits for binary classification

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            batch_labels = batch_labels.to(device)  

            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            logits = outputs.logits

            for logit_idx in range(len(logits)):
                for i in range(0, num_labels, num_classes):
                    subarray_logits = logits[logit_idx]
                    subarray_logits = subarray_logits[i: i + num_classes]

                    best_class = np.argmax(subarray_logits.detach().numpy(), axis=-1)
                    subarray_logits[best_class] = 1
                    for arg in range(len(subarray_logits)):
                        if arg == best_class:
                            continue
                        subarray_logits[arg] = 0
                    
                    subarray_labels = batch_labels[logit_idx][i: i + num_classes]

                    #Calculating accuracy
                    temp = int((i)/num_classes)
                    label_sample_counts[labels[temp]] += 1.0
                    if subarray_labels[best_class] == 1:
                        label_accuracies[labels[temp]] += 1.0

    for label in labels:
        label_accuracy = label_accuracies[label] / label_sample_counts[label]
        print(f"Accuracy for {label}: {label_accuracy:.4f}")

if __name__ == "__main__":
    main()