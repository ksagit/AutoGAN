import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import faiss


class NeighborDiscriminator(nn.Module):

    def __init__(
            self,
            X: torch.Tensor,
            K: float = 1,
            nlist: int = 100,
            nprobe: int = 100,
            k: int = 10,  # number of neighbors to check

    ):
        super(NeighborDiscriminator, self).__init__()
        self.X = X.view(X.shape[0], -1)
        self.w = nn.Parameter(torch.randn(X.shape[0], 1) * 10)
        self.K = K

        self.n, self.d = self.X.shape
        self.nlist = nlist
        self.nprobe = nprobe
        self.k = k

    def w_prime(self):
        w_prime = -(self.w - torch.max(F.relu(self.w)))
        w_prime = torch.sqrt(w_prime / self.K)
        return w_prime

    def X_w(self):
        X = self.X.data.cpu().numpy()
        w_prime = self.w_prime().data.cpu().numpy()

        return np.hstack([X, w_prime]).astype('float32')

    def get_index(self):
        index = faiss.IndexFlatL2(self.d + 1)
        index.add(self.X_w())
        return index

    def get_approximated_neighbors(self, X_tilde):
        X_tilde = X_tilde.view(X_tilde.shape[0], -1)
        self.X_tilde_padded = np.hstack(
            [
                X_tilde.data.cpu().numpy(),
                np.zeros((X_tilde.shape[0], 1))
            ]
        ).astype('float32')

        index = self.get_index()

        D, I = index.search(self.X_tilde_padded, self.k)
        return D, I

    def get_actual_distances(self, D, I):
        D, I = torch.from_numpy(D), torch.from_numpy(I)

        local_w_adjustments = self.w_prime()[I].squeeze(2) ** 2
        D_actual_squared = F.relu(D - local_w_adjustments)

        D_actual = torch.sqrt(D_actual_squared)
        return D_actual, I

    def get_maximal_neighbors(self, D_actual, I):
        f_i_values = self.w[I].squeeze(2) - self.K * D_actual
        minimal_distance_indices = torch.argmax(f_i_values, axis=1)
        return torch.Tensor([row[index] for row, index in zip(I, minimal_distance_indices)])

    def forward(self, X_tilde):
        D, I = self.get_approximated_neighbors(X_tilde)
        D, I = self.get_actual_distances(D, I)
        return dis.get_maximal_neighbors(D, I)


class QuantizedNeighborDiscriminator(NeighborDiscriminator):

    def __init__(self, *args, **kwargs):
        super(QuantizedNeighborDiscriminator, self).__init__(*args, **kwargs)

    def get_index(self):
        quantizer = faiss.IndexFlatL2(self.d + 1)  # the other index
        index = faiss.IndexIVFFlat(quantizer, self.d + 1, self.nlist)

        xb = self.X_w()
        assert not index.is_trained
        index.train(xb)
        assert index.is_trained
        index.add(xb)
        return index