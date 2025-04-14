import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import esm


# -----------------------------------------
# Module: model_utils.py
# -----------------------------------------

def load_esm2_model():
    """
    Loads the pretrained ESM2 model, alphabet, and batch converter.
    Returns:
        esm_model: The ESM2 model.
        alphabet: The model's alphabet.
        batch_converter: Function to tokenize sequences.
    """
   
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, alphabet, batch_converter

class ESM2_CARTModel(nn.Module):
    def __init__(self, esm_model, hidden_size, num_labels=1):
        """
        Custom model wrapping the ESM2 backbone with a classification/regression head.
        Args:
            esm_model: Pretrained ESM2 model.
            hidden_size: Embedding dimension (should match esm_model.embed_dim).
            num_labels: Number of output labels (1 for regression; change as needed).
        """
        super(ESM2_CARTModel, self).__init__()
        self.esm_model = esm_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, tokens):
        # Obtain the final-layer representations from ESM2.
        outputs = self.esm_model(tokens, repr_layers=[-1])
        token_reps = outputs["representations"][-1]  # Shape: (batch_size, seq_len, hidden_size)
        # Average pool over sequence length.
        pooled = token_reps.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits.squeeze(-1)

# -----------------------------------------
# Module: data_utils.py
# -----------------------------------------

class CARTDataset(Dataset):
    """
    Dataset for CAR-T related sequences and corresponding labels.
    Each sample is a tuple: (sequence, label).
    """
    def __init__(self, sequences, labels, alphabet):
        self.sequences = sequences
        self.labels = labels
        self.alphabet = alphabet  # Used for tokenization
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch, batch_converter):
    """
    Collate function to tokenize sequences using ESM2's batch converter.
    Args:
        batch: List of tuples (sequence, label).
        batch_converter: Function to convert raw sequences into tokenized format.
    Returns:
        tokens: Tensor of tokenized sequences.
        labels: Tensor of labels.
    """
    sequences, labels = zip(*batch)
    # Prepare data as list of tuples (id, sequence)
    data = [(i, seq) for i, seq in enumerate(sequences)]
    _, tokens = batch_converter(data)
    return tokens, torch.tensor(labels, dtype=torch.float32)

# -----------------------------------------
# Module: train_utils.py
# -----------------------------------------

def train_model(model, dataloader, optimizer, criterion, num_epochs):
    """
    Trains the model using the provided dataloader, optimizer, and loss criterion.
    Args:
        model: The model to train.
        dataloader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        num_epochs: Number of epochs to train.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for tokens, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# -----------------------------------------
# Main Script
# -----------------------------------------

def main():
    # 1. Load the pretrained ESM2 model and batch converter.
    esm_model, alphabet, batch_converter = load_esm2_model()
    
    # 2. Instantiate the custom model.
    hidden_size = esm_model.embed_dim
    model = ESM2_CARTModel(esm_model, hidden_size, num_labels=1)  # Example for regression
    
    # 3. Prepare your dataset.
    # Replace the dummy sequences and labels with your CAR-T data.
    sequences = [
        "MGSSHHHHHHSSGLVPRGSHM", 
        "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDICNDVLELLDK"
    ] * 10  # Dummy list of sequences
    labels = [1.0, 0.0] * 10  # Dummy labels (e.g., binding affinity)
    
    dataset = CARTDataset(sequences, labels, alphabet)
    # Use a lambda to pass batch_converter to collate_fn.
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, 
                            collate_fn=lambda batch: collate_fn(batch, batch_converter))
    
    # 4. Set up the optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()  # For regression; switch to CrossEntropyLoss for classification
    
    # 5. Train the model.
    num_epochs = 3  # Adjust as needed.
    train_model(model, dataloader, optimizer, criterion, num_epochs)
    
    # 6. Save the trained model.
    torch.save(model.state_dict(), "esm2_cart_model.pt")
    print("Model training complete and saved.")

    # Verify the ESM package version
    print(fair_esm.__version__)

if __name__ == '__main__':
    main()
