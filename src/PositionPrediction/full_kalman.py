import numpy as np

class SoccerKalmanFilter:
    """
    Filtro di Kalman per il tracking dei giocatori (SoccerNet).
    Stato: [x, y, w, h, vx, vy, vw, vh]
    Osservazione: [x, y, w, h] (da YOLO)
    """
    def __init__(self, initial_box):
        # Stato: [x, y, w, h, vx, vy, vw, vh]
        self.state = np.zeros(8)
        self.state[:4] = initial_box
        
        # Matrice di Transizione (Constant Velocity Model)
        # x_t = x_{t-1} + vx_{t-1} * dt
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0  # Assumiamo dt = 1 frame
            
        # Matrice di Osservazione (mappa lo stato in quello che YOLO vede)
        self.H = np.eye(4, 8)
        
        # Matrice di Covarianza dello Stato (Incertezza iniziale)
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0 # Maggiore incertezza sulla velocità iniziale
        
        # Rumore di Processo (quanto ci aspettiamo che il modello CV sbagli)
        self.Q = np.eye(8)
        self.Q[:4] *= 0.1    # Incertezza bassa sulla posizione
        self.Q[4:] *= 0.01   # Incertezza bassissima sulla velocità (inerzia)

        # Rumore di Misura (quanto ci fidiamo di YOLO)
        self.R = np.eye(4) * 1.0

    def predict(self):
        """
        Passo di Predizione: Proietta lo stato in avanti.
        Aumenta l'incertezza (P) se non ci sono misure.
        """
        # x = F * x
        self.state = np.dot(self.F, self.state)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:4]

    def update(self, measurement):
        """
        Passo di Correzione: YOLO ha visto il giocatore.
        Riduce l'incertezza basandosi sulla misura reale.
        """
        # Calcolo del Guadagno di Kalman (K)
        # S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # K = P * H^T * inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Aggiornamento dello stato
        # y = z - H * x (residuo)
        y = measurement - np.dot(self.H, self.state)
        # x = x + K * y
        self.state = self.state + np.dot(K, y)
        
        # Aggiornamento della covarianza
        # P = (I - K * H) * P
        I = np.eye(8)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_uncertainty(self):
        """
        Restituisce un valore scalare che rappresenta l'incertezza attuale.
        Utile per decidere quando smettere di predire e "killare" il track.
        """
        return np.trace(self.P[:2, :2]) # Traccia della covarianza x,y