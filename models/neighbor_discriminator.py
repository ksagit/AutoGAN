import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import faiss
from torch.autograd import Function
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def swig_ptr_from_FloatTensor(x: torch.Tensor):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(x.storage().data_ptr())


class NeighborDiscriminatorFunction(Function):

    @staticmethod
    def forward(ctx, input, model):
        D, I = model.get_approximated_neighbor_activations(input)
        D_actual, I_actual = model.get_actual_distances(D, I)
        D, I = model.get_maximal_neighbor_activations(D_actual, I_actual)

        neighbors = model.X[I].cuda()
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

class DummyDistanceFunction(Function):

    @staticmethod
    def forward(ctx, distance_hints):
        pass


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
        return w_prime.cuda()

    def get_X_w(self):
        X = self.X.data
        w_prime = self.w_prime
        return torch.cat([X, w_prime], 1)

    def update_index(self):
        self.w_prime = self.get_w_prime()
        self.X_w = self.get_X_w()

        gpu_resource = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(gpu_resource, self.d + 1)
        
        X_w_casted = swig_ptr_from_FloatTensor(self.X_w)
        index.add_c(self.n, X_w_casted)
        self.index = index

      
    def get_approximated_neighbor_activations(self, X_tilde):
        """Get the nearest neighbors in \|x_i \oplus \sqrt{w_i'/K} - x \oplus 0\| """
        m = X_tilde.shape[0]
        X_tilde = X_tilde.view(m, -1)
        X_tilde_padded = torch.cat(
            [
                X_tilde,
                torch.zeros((m, 1)).cuda()
            ],
            1
        )
        
        X_tilde_padded_casted = swig_ptr_from_FloatTensor(X_tilde_padded)

        # make space for indexes
        distances = torch.zeros((m, self.k), dtype=torch.float32).cuda()
        indexes = torch.zeros((m, self.k), dtype=torch.long).cuda()
        
        self.index.search_c(
            X_tilde.shape[0], X_tilde_padded_casted, self.k,
            swig_ptr_from_FloatTensor(distances), swig_ptr_from_LongTensor(indexes)
        )

        logger.info("If I don't print %s it doesn't work" % repr(indexes))  # fuck fuck fuck
        return distances, indexes  

    def get_maximal_neighbor_activations(self, D_actual, I):
        """Use the \|x_i - x\| to get the w_i - K \|x_i - x\|, and find the max and argmax over i"""
        neighbor_activations = self.w[I].squeeze(2) - self.K * D_actual
        maximal_neighbor_activation_indices = torch.argmax(neighbor_activations, axis=1, keepdim=True)

        maximal_neighbor_activations = neighbor_activations.gather(1, maximal_neighbor_activation_indices)
        return maximal_neighbor_activations.squeeze(1)
    

    def forward(self, X_tilde):
        X_tilde = X_tilde.view(X_tilde.shape[0], -1)
        with torch.no_grad():
            _, maximal_neighbor_activation_indices = self.get_approximated_neighbor_activations(X_tilde)

        neighbor_vectors = self.X[maximal_neighbor_activation_indices]  # batchsize x k x img size
        differences = (neighbor_vectors - X_tilde.unsqueeze(1))  # batchsize x k x img size - batchsize x 1 x img size
        distances = torch.norm(differences, dim=2)

        return self.get_maximal_neighbor_activations(distances, maximal_neighbor_activation_indices)


    def project_weights(self):
        with torch.no_grad():
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