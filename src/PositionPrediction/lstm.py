import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMPredictor(nn.Module):
    """
    Rete Ricorrente LSTM per la predizione di traiettorie.
    Input: Sequenza di N box [x, y, w, h] -> (Batch, Seq_Len, 4)
    Output: Box al frame successivo -> (Batch, 4)
    """
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout_prob=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Strato LSTM: Elabora la sequenza temporale
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Strato Dropout per Monte Carlo Dropout (Incertezza)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Head di predizione
        self.fc = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, 4)
        
        # out: (Batch, Seq_Len, hidden_dim)
        # _ : (h_n, c_n) stati interni che qui non usiamo per la predizione singola
        out, _ = self.lstm(x)
        
        # Prendiamo solo l'output dell'ultimo timestep (l'ultimo frame della sequenza)
        last_out = out[:, -1, :]
        
        # Applichiamo dropout e proiezione finale
        last_out = self.dropout(last_out)
        prediction = self.fc(last_out)
        
        return prediction

class LSoccerManager:
    def __init__(self, hidden_dim=64, device='cpu'):
        self.device = device
        self.model = LSTMPredictor(hidden_dim=hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def predict_with_uncertainty(self, sequence, n_iter=10):
        """
        Esegue multiple inferenze con Dropout attivo per stimare l'incertezza.
        sequence shape: (Seq_Len, 4)
        """
        self.model.train() # Dropout attivo
        predictions = []
        
        # (1, Seq_Len, 4)
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_iter):
                pred = self.model(input_tensor)
                predictions.append(pred.cpu().numpy()[0])
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.var(axis=0).sum()
        
        return mean_pred, uncertainty

    def train_step(self, batch_x, batch_y):
        """
        batch_x: (Batch, Seq_Len, 4)
        batch_y: (Batch, 4)
        """
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        loss.backward()
        self.optimizer.step()
        return loss.item()