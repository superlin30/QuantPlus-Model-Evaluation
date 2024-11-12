import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import gc

# Dataset Class for Stock Data
class StockDataset_by_year(Dataset):
    def __init__(self, base_dir, start_year, end_year, task='regression'):
        self.base_dir = base_dir
        self.years = list(range(start_year, end_year + 1))
        self.task = task
        self.data = []
        self.labels = []
        
        # Load data with progress bar
        for year in tqdm(self.years, desc="Loading Data by Year"):
            data_file = os.path.join(base_dir, f"{year}_df_data.feather")
            label_file = os.path.join(base_dir, f"{year}_real_RET.feather")
            
            # Read data and labels
            X = pd.read_feather(data_file)
            y = pd.read_feather(label_file)
            
            # Drop 'permno' and 'DATE' columns, keep the rest as features
            X = X.drop(columns=['permno', 'DATE'])
            
            # Convert labels to 0 or 1 if task is classification
            if self.task == 'classification':
                y = (y > 0).astype(int)
            
            # Convert to numpy arrays and store
            self.data.append(X.values)
            self.labels.append(y.values)
        
        # Convert data and labels to single arrays
        self.data = torch.tensor(np.vstack(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.vstack(self.labels), dtype=torch.float32)
        
        # Print dataset information
        print(f"Data loaded from {start_year} to {end_year}.")
        print(f"Total samples: {len(self.labels)}")
        print(f"Feature dimension: {self.data.shape[1]}")
        print(f"Task: {self.task}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Function to create dataloaders
def create_dataloaders(base_dir, start_year_train, start_year_test, end_year_test, task='regression', batch_size=32):
    print("Constructing training dataset...")
    train_dataset = StockDataset_by_year(base_dir, start_year_train, start_year_test - 1, task)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Constructing testing dataset...")
    test_dataset = StockDataset_by_year(base_dir, start_year_test, end_year_test, task)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# NN1 Model Definition
class NN1(nn.Module):
    def __init__(self, input_dim, hidden_neurons=128):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neurons)
        self.out = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.out(x)

# NN2 Model Definition
class NN2(nn.Module):
    def __init__(self, input_dim, hidden_neurons=128, dropout_rate=0.3):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neurons)
        self.bn1 = nn.BatchNorm1d(hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons // 2)
        self.bn2 = nn.BatchNorm1d(hidden_neurons // 2)
        self.out = nn.Linear(hidden_neurons // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.out(x)

# NN3 Model Definition
class NN3(nn.Module):
    def __init__(self, input_dim, hidden_neurons1=128, hidden_neurons2=64, hidden_neurons3=32, dropout_rate1=0.3, dropout_rate2=0.4, dropout_rate3=0.5):
        super(NN3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neurons1)
        self.bn1 = nn.BatchNorm1d(hidden_neurons1)
        self.fc2 = nn.Linear(hidden_neurons1, hidden_neurons2)
        self.bn2 = nn.BatchNorm1d(hidden_neurons2)
        self.fc3 = nn.Linear(hidden_neurons2, hidden_neurons3)
        self.bn3 = nn.BatchNorm1d(hidden_neurons3)
        self.out = nn.Linear(hidden_neurons3, 1)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.dropout3 = nn.Dropout(dropout_rate3)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return self.out(x)


# Define 1D CNN model
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes=1, dropout_rate=0.3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (input_dim // 8), 64)  # Calculate the dimension after pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension, transform to [batch_size, 1, input_dim]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)

# Define standard RNN model
class StandardRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(StandardRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        # Change input shape to (batch_size, seq_length=1, input_dim)
        x = x.unsqueeze(1)  # Add a time dimension seq_length=1
        x, _ = self.rnn(x)  # Output x shape is (batch_size, seq_length, hidden_dim)
        x = x[:, -1, :]  # Select the output of the last time step
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        return self.out(x)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a time dimension seq_length=1
        x, _ = self.lstm(x)  # x shape is (batch_size, seq_length, hidden_dim)
        x = x[:, -1, :]  # Select the output of the last time step
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        return self.out(x)

# Define a simplified Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Input embedding layer, maps input dimension to d_model
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        # x shape is (batch_size, input_dim) or (batch_size, seq_length, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a time dimension, making x shape (batch_size, seq_length=1, input_dim)
        
        x = self.embedding(x)  # Map input to d_model dimension
        x = x.permute(1, 0, 2)  # Transform to (seq_length, batch_size, d_model) for Transformer
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Transform back to (batch_size, seq_length, d_model)
        x = x[:, -1, :]  # Select the output of the last time step
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)




# Simple GNN Model Definition
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(SimpleGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)

def transform_to_graph_data(X_batch, y_batch):
    batch_size = X_batch.size(0)
    num_nodes = X_batch.size(0)  
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous() 
    batch = torch.arange(batch_size, dtype=torch.long).repeat_interleave(1)  
    data = Data(x=X_batch, edge_index=edge_index, y=y_batch, batch=batch)
    return data

def train_model_once_gnn(model, train_loader, test_loader, input_dim, num_epochs=50, lr=0.001, patience=10, save_base_path=None, use_scheduler=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Optionally add a learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    results = {'train_loss': [], 'test_loss': [], 'test_r2': [], 'learning_rate': []}


    model_folder = os.path.join(save_base_path, model.__class__.__name__)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    total_start_time = time.time()


    epoch_iterator = tqdm(range(num_epochs), desc='Training Epochs', unit='epoch')
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

  
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit='batch')
        for X_batch, y_batch in train_loader_iter:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
 
            graph_data = transform_to_graph_data(X_batch, y_batch).to(device)
            

            outputs = model(graph_data.x, graph_data.edge_index, graph_data.batch)
            loss = criterion(outputs, graph_data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        results['train_loss'].append(train_loss)

        model.eval()
        test_loss = 0.0
        y_test_pred_list = []
        y_test_list = []
      
        test_loader_iter = tqdm(test_loader, desc=f"Epoch {epoch+1} Testing", leave=False, unit='batch')
        with torch.no_grad():
            for X_batch, y_batch in test_loader_iter:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
         
                graph_data = transform_to_graph_data(X_batch, y_batch).to(device)
                
          
                outputs = model(graph_data.x, graph_data.edge_index, graph_data.batch)
                loss = criterion(outputs, graph_data.y)
                test_loss += loss.item() * X_batch.size(0)
                y_test_pred_list.append(outputs.cpu().numpy())
                y_test_list.append(y_batch.cpu().numpy())
        
        test_loss /= len(test_loader.dataset)
        y_test_pred = np.concatenate(y_test_pred_list).ravel()
        y_test = np.concatenate(y_test_list).ravel()
        test_r2 = calculate_r2_oos(y_test, y_test_pred)
        
        results['test_loss'].append(test_loss)
        results['test_r2'].append(test_r2)

        # Log the current learning rate if scheduler is used
        if scheduler:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            results['learning_rate'].append(current_lr)

      
        epoch_iterator.set_postfix({
            'Train Loss': f"{train_loss:.6f}", 
            'Test Loss': f"{test_loss:.6f}", 
            'Test R²': f"{test_r2:.6f}",
            'LR': current_lr if use_scheduler else lr
        })

        # Use scheduler if applicable
        if scheduler:
            scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    total_training_time = time.time() - total_start_time


    if best_model_state:
        model.load_state_dict(best_model_state)
        model_save_path = os.path.join(model_folder, f"one_loop_best_model.pth")
        torch.save(model.state_dict(), model_save_path)

     
        history_save_path = os.path.join(model_folder, "one_loop_training_history.txt")
        with open(history_save_path, 'w') as f:
            f.write("Epoch\tTrain Loss\tTest Loss\tTest R²\tLR\n")
            for epoch in range(len(results['train_loss'])):
                f.write(f"{epoch+1}\t{results['train_loss'][epoch]:.6f}\t{results['test_loss'][epoch]:.6f}\t{results['test_r2'][epoch]:.6f}\t{results['learning_rate'][epoch] if use_scheduler else lr:.6f}\n")

        return model_save_path, results['test_r2'][-1], total_training_time
    else:
        return None, results['test_r2'][-1], total_training_time


def train_model_once(model, train_loader, test_loader, input_dim, num_epochs=50, lr=0.001, patience=10, save_base_path=None, use_scheduler=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    results = {'train_loss': [], 'test_loss': [], 'test_r2': [], 'learning_rate': []}

    model_folder = os.path.join(save_base_path, model.__class__.__name__)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    total_start_time = time.time()
    epoch_iterator = tqdm(range(num_epochs), desc='Training Epochs', unit='epoch')
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit='batch')
        for X_batch, y_batch in train_loader_iter:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        results['train_loss'].append(train_loss)

        model.eval()
        test_loss = 0.0
        y_test_pred_list = []
        y_test_list = []
        test_loader_iter = tqdm(test_loader, desc=f"Epoch {epoch+1} Testing", leave=False, unit='batch')
        with torch.no_grad():
            for X_batch, y_batch in test_loader_iter:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
                y_test_pred_list.append(outputs.cpu().numpy())
                y_test_list.append(y_batch.cpu().numpy())
        test_loss /= len(test_loader.dataset)
        y_test_pred = np.concatenate(y_test_pred_list).ravel()
        y_test = np.concatenate(y_test_list).ravel()
        test_r2 = calculate_r2_oos(y_test, y_test_pred)
        results['test_loss'].append(test_loss)
        results['test_r2'].append(test_r2)

        if scheduler:
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            results['learning_rate'].append(current_lr)
        epoch_iterator.set_postfix({'Train Loss': f"{train_loss:.6f}", 'Test Loss': f"{test_loss:.6f}", 'Test R²': f"{test_r2:.6f}", 'LR': current_lr if use_scheduler else lr})

        if scheduler:
            scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    total_training_time = time.time() - total_start_time
    if best_model_state:
        model.load_state_dict(best_model_state)
        model_save_path = os.path.join(model_folder, f"one_loop_best_model.pth")
        torch.save(model.state_dict(), model_save_path)
        history_save_path = os.path.join(model_folder, "one_loop_training_history.txt")
        with open(history_save_path, 'w') as f:
            f.write("Epoch\tTrain Loss\tTest Loss\tTest R²\tLR\n")
            for epoch in range(len(results['train_loss'])):
                f.write(f"{epoch+1}\t{results['train_loss'][epoch]:.6f}\t{results['test_loss'][epoch]:.6f}\t{results['test_r2'][epoch]:.6f}\t{results['learning_rate'][epoch] if use_scheduler else lr:.6f}\n")
        return model_save_path, results['test_r2'][-1], total_training_time
    else:
        return None, results['test_r2'][-1], total_training_time
    

