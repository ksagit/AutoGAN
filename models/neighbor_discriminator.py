import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import faiss
from torch.autograd import Function

import logging

logger = logging.getLogger(__name__)

class NeighborDiscriminatorFunction(Function):

    @staticmethod
    def forward(ctx, input, model):
        D, I = model.get_approximated_neighbor_activations(input)
        D_actual, I_actual = model.get_actual_distances(D, I)
        D, I = model.get_maximal_neighbor_activations(D_actual, I_actual)

        neighbors = model.X[I]
        D_actual_at_maxes_mask = torch.cat(
            [
                (I_actual[:, col] == I).unsqueeze(1)
                for col in range(I_actual.shape[1])
            ], dim=1
        )
        D_actual_at_maxes = D_actual.masked_select(D_actual_at_maxes_mask)
        scaled_stabilized_D_actual_at_maxes = D_actual_at_maxes / model.K + 1e-5
        ctx.save_for_backward(input, neighbors, scaled_stabilized_D_actual_at_maxes)
        return D, I

    @staticmethod
    def backward(ctx, grad_output, _grad_indices):
        input, neighbors, scaled_D_actual = ctx.saved_tensors
        return (neighbors - input) / scaled_D_actual.unsqueeze(1), None


class NeighborDiscriminator(nn.Module):

    def __init__(
            self,
            X: torch.Tensor,
            K: float = 1,
            nlist: int = 100,
            nprobe: int = 10,
            k: int = 10,  # number of neighbors to check
            eta: float = .1

    ):
        super(NeighborDiscriminator, self).__init__()
        self.X = X.view(X.shape[0], -1)
        self.w = nn.Parameter(torch.zeros(X.shape[0], 1))
        self.K = K

        self.n, self.d = self.X.shape
        self.nlist = nlist
        self.nprobe = nprobe
        self.k = k

        self.eta = eta

        # also initializes self.X_w, self.w_prime
        self.update_index()

    def get_w_prime(self):
        w_prime = -(self.w - torch.max(self.w))
        w_prime = torch.sqrt(w_prime / self.K)
        return w_prime

    def get_X_w(self):
        X = self.X.data.cpu().numpy()
        w_prime = self.w_prime.data.cpu().numpy()
        return np.hstack([X, w_prime]).astype('float32')

    def update_index(self):
        self.w_prime = self.get_w_prime()
        self.X_w = self.get_X_w()
        index = faiss.IndexFlatL2(self.d + 1)
        index.add(self.X_w)
        self.index = index

    def get_approximated_neighbor_activations(self, X_tilde):
        """Get the nearest neighbors in \|x_i \oplus \sqrt{w_i'/K} - x \oplus 0\| """
        X_tilde = X_tilde.view(X_tilde.shape[0], -1)
        X_tilde_padded = np.hstack(
            [
                X_tilde.data.cpu().numpy(),
                np.zeros((X_tilde.shape[0], 1))
            ]
        ).astype('float32')
        D, I = self.index.search(X_tilde_padded, self.k)
        return D, I

    def get_actual_distances(self, D, I):
        """Adjust the approximated neighbor activations to get the \|x_i - x\|"""
        D, I = torch.from_numpy(D), torch.from_numpy(I)

        local_w_adjustments = self.w_prime[I].squeeze(2)
        D_actual_squared = F.relu(D - local_w_adjustments ** 2)

        D_actual = torch.sqrt(D_actual_squared)
        return D_actual, I

    def get_maximal_neighbor_activations(self, D_actual, I):
        """Use the \|x_i - x\| to get the w_i - K \|x_i - x\|, and find the max and argmax over i"""
        neighbor_activations = self.w[I].squeeze(2) - self.K * D_actual
        maximal_neighbor_activation_indices = torch.argmax(neighbor_activations, axis=1)
        D_I = torch.Tensor(
            [
                (dist_row[index], index_row[index])
                for dist_row, index_row, index in zip(neighbor_activations, I, maximal_neighbor_activation_indices)
            ]
        )
        return D_I[:, 0], D_I[:, 1].long()

    def forward(self, X_tilde):
        return NeighborDiscriminatorFunction.apply(X_tilde, self)

    def accum_grads(self, X_tilde):
        _, I = self(X_tilde)
        counts = torch.zeros(self.n)
        for label in I:
            counts[label] += 1
        update_indices = torch.where(counts > 0)
        self.w.grad = counts.unsqueeze(1)
        return update_indices

    def project_weights(self, update_indices):
        with torch.no_grad():
            self.update_index()
            self.w.data[update_indices] = self.forward(self.X[update_indices])[0].unsqueeze(1)
            self.w -= self.w.mean()
            self.update_index()

class QuantizedNeighborDiscriminator(NeighborDiscriminator):
    """Doesn't work very well..."""

    def update_index(self):
        self.w_prime = self.get_w_prime()
        self.X_w = self.get_X_w()

        quantizer = faiss.IndexFlatL2(self.d + 1)
        index = faiss.IndexIVFFlat(quantizer, self.d + 1, self.nlist)
        index.train(self.X_w)
        index.add(self.X_w)
        self.index = index