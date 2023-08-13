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

from data.HIV_data import load_HIV


def main():
    labels, df = load_HIV()
    print(labels)
    print("Data Loaded and Preprocessing Finished")

    num_classes = 2
    num_labels = len(df.columns) - 1
    max_sequence_length = df['smiles'].apply(len).max()
    #print(vocab_size)
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001



    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print("Train Test Split Finished")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_labels)
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

    label_accuracies = {label: 0.0 for label in labels}
    label_sample_counts = {label: 0 for label in labels}

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
            logits = outputs.logits

            batch_loss = 0.0
            for logit_idx in range(len(logits)):
                logit_loss = 0.0
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

                    loss = criterion(subarray_logits, subarray_labels)
                    logit_loss += loss

                    #Calculating accuracy
                    temp = int((i)/num_classes)
                    label_sample_counts[labels[temp]] += 1.0
                    if subarray_labels[best_class] == 1:
                        label_accuracies[labels[temp]] += 1.0
                batch_loss += logit_loss
            print(f"Batch Loss: {batch_loss}")

            for label in labels:
                print(f"Accuracy for {label}: {label_accuracies[label]/label_sample_counts[label]}")
            
            
            
            batch_loss.backward()
            optimizer.step()

            label_accuracies = {label: 0.0 for label in labels}
            label_sample_counts = {label: 0 for label in labels}

    torch.save(model.state_dict(), os.path.join(os.getcwd(), 'models', 'tox21.pth'))

if __name__ == "__main__":
    main()