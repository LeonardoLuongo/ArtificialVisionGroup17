import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPPredictor(nn.Module):
    """
    Multi-Layer Perceptron per la predizione della posizione.
    Input: Finestra di N frame precedenti [x, y, w, h] * N
    Output: Box al frame successivo [x, y, w, h]
    """
    def __init__(self, window_size=10, hidden_dim=128, dropout_prob=0.2):
        super(MLPPredictor, self).__init__()
        
        input_dim = window_size * 4  # 4 coordinate (x,y,w,h) per ogni frame
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            
            nn.Linear(hidden_dim // 2, 4) # Output: x, y, w, h
        )
        
    def forward(self, x):
        # x shape: (batch_size, window_size * 4)
        return self.network(x)

class MLPSoccerManager:
    def __init__(self, window_size=10, device='cpu'):
        self.window_size = window_size
        self.device = device
        self.model = MLPPredictor(window_size=window_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def normalize(self, box, img_w=1920, img_h=1080):
        """ Normalizza le coordinate tra 0 e 1 per facilitare l'apprendimento """
        return [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]

    def denormalize(self, box, img_w=1920, img_h=1080):
        return [box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h]

    def predict_with_uncertainty(self, sequence, n_iter=10):
        """
        Esegue N inferenze con Dropout attivo. 
        La varianza dei risultati indica l'incertezza del modello.
        """
        self.model.train() # Mantieni dropout attivo anche in fase di test
        predictions = []
        
        # Converti sequenza in tensore (1, window_size * 4)
        input_tensor = torch.FloatTensor(sequence).view(1, -1).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_iter):
                pred = self.model(input_tensor)
                predictions.append(pred.cpu().numpy()[0])
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.var(axis=0).sum() # Somma delle varianze di x,y,w,h
        
        return mean_pred, uncertainty

    def train_step(self, batch_x, batch_y):
        """
        batch_x: (batch, window_size * 4)
        batch_y: (batch, 4)
        """
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        loss.backward()
        self.optimizer.step()
        return loss.item()