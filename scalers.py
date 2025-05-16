from sklearn.preprocessing import StandardScaler
import numpy as np
# Definition of CustomScaler that scales all columns except the ones in columns_to_exclude (mole fraction here)
class CustomScaler:
    def __init__(self, Embedding_BERT):
        self.scaler = StandardScaler()
        self.columns_to_exclude = [1]
        self.Embedding_BERT = Embedding_BERT
        self.feature_dim = 2 + Embedding_BERT  # <-- Add this line!


    def fit_transform(self, X, y=None):
        X = np.copy(X)
        N, n_comp, dim = X.shape
        assert dim == self.feature_dim, f"Expected dim={self.feature_dim}, but got {dim}"
        X_reshaped = X.reshape(N * n_comp, dim)
        columns_to_scale = [i for i in range(dim) if i not in self.columns_to_exclude]
        X_reshaped[:, columns_to_scale] = self.scaler.fit_transform(X_reshaped[:, columns_to_scale])
        return X_reshaped.reshape(N, n_comp, dim)

    def transform(self, X, y=None):
        X = np.copy(X)
        N, n_comp, dim = X.shape
        assert dim == self.feature_dim, f"Expected dim={self.feature_dim}, but got {dim}"
        X_reshaped = X.reshape(N * n_comp, dim)
        columns_to_scale = [i for i in range(dim) if i not in self.columns_to_exclude]
        X_reshaped[:, columns_to_scale] = self.scaler.transform(X_reshaped[:, columns_to_scale])
        return X_reshaped.reshape(N, n_comp, dim)

    def inverse_transform(self, X):
        X = np.copy(X)
        N, n_comp, dim = X.shape
        assert dim == self.feature_dim, f"Expected dim={self.feature_dim}, but got {dim}"
        X_reshaped = X.reshape(N * n_comp, dim)
        columns_to_scale = [i for i in range(dim) if i not in self.columns_to_exclude]
        X_reshaped[:, columns_to_scale] = self.scaler.inverse_transform(X_reshaped[:, columns_to_scale])
        return X_reshaped.reshape(N, n_comp, dim)
