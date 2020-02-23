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


class CustomBatchNorm(nn.Module):

    def __init__(self, momentum=.9):
        super().__init__()
        self.running_mean = 0
        self.running_variance = 1
        self.eps = 2e-5
        self.momentum = momentum

    def forward(self, x, importance=1):
        assert (x.dim() == 2 and x.shape[1] == 1)
        assert (0. <= importance <= 1)

        # update statistics
        update_size = (1 - self.momentum) * importance

        batch_mean = torch.mean(x).detach()
        batch_variance = torch.var(x).detach()

        self.running_mean = (1 - update_size) * self.running_mean + update_size * batch_mean
        self.running_variance = (1 - update_size) * self.running_variance + update_size * batch_variance

        # normalize and return x
        return (x - self.running_mean) / (self.running_variance + self.eps).sqrt()

def torch_pairwise_distances(X, Y):
    return torch.sqrt(torch.abs(
        torch.sum(X * X, dim=1, keepdim=True)
        - 2 * torch.matmul(X, Y.transpose(0,1))
        + torch.sum(Y * Y, dim=1, keepdim=True).transpose(0,1)
    ))

class RetardedNeighborDiscriminator(nn.Module):

    def __init__(
        self,
        X,
        K
    ):
        super().__init__()
        self.X = X.view(X.shape[0], -1)
        self.w = nn.Parameter(torch.zeros(X.shape[0], 1))
        self.K = K

    def forward(self, X_tilde):
        X_tilde = X_tilde.view(X_tilde.shape[0], -1)

        pairwise_distances = torch_pairwise_distances(self.X, X_tilde)
        exact_neighbor_activations = -self.K * pairwise_distances + self.w
        exact_maximal_neighbor_activations, _ = torch.max(exact_neighbor_activations, axis=0)

        ret = exact_maximal_neighbor_activations.unsqueeze(1)
        assert(ret.shape[1] == 1)
        assert(ret.dim() == 2)
        return ret

class NeighborDiscriminator(nn.Module):

    def __init__(
            self,
            X: torch.Tensor,
            K: float = 750,
            nlist: int = 100,
            nprobe: int = 10,
            k: int = 16,  # number of neighbors to check
            eta: float = .1

    ):
        super(NeighborDiscriminator, self).__init__()
        self.X = X.view(X.shape[0], -1)
        self.w = nn.Parameter(torch.zeros(X.shape[0], 1))
        self.bn = nn.BatchNorm1d(num_features=1)

        nn.init.xavier_uniform(self.w.data, 1.)

        self.K = K

        self.n, self.d = self.X.shape
        self.nlist = nlist
        self.nprobe = nprobe
        self.k = k
        self.bn = CustomBatchNorm()

        self.eta = eta

        self.update_index()

    def get_w_prime(self):
        w_prime = -(self.w - torch.max(self.w))
        w_prime = w_prime / self.K
        return w_prime.cuda()

    def get_X_w(self):
        X = self.X.data
        w_prime = self.w_prime
        return torch.cat([X, w_prime], 1)

    def alloc_index(self):
        self.gpu_resource = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatL2(self.gpu_resource, self.d + 1)

    def free_index(self):
        del self.gpu_resource
        del self.index

    def update_index(self):
        if not(hasattr(self, 'index')) or not(hasattr(self, 'gpu_resource')):
            self.alloc_index()

        self.w_prime = self.get_w_prime()
        self.X_w = self.get_X_w()

        self.index.reset()
        
        X_w_casted = swig_ptr_from_FloatTensor(self.X_w)
        self.index.add_c(self.n, X_w_casted)

      
    def get_approximated_neighbor_activations(self, X_tilde, noise=0):
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

        if noise != 0.0:
            noise_frac = int(m * noise)
            indices_to_noise = torch.randperm(m) < noise_frac

            noisy_labels = torch.randint(high=50000, size=(noise_frac,)).cuda()
            noisy_labels = noisy_labels.repeat((self.k, 1)).T

            # assert(indexes[indices_to_noise].shape == noisy_labels.shape)
            indexes[indices_to_noise] = noisy_labels

        return distances, indexes

    def get_maximal_neighbor_activations(self, distances, maximal_neighbor_activation_indices):
        """Use the \|x_i - x\| to get the w_i - K \|x_i - x\|, and find the max and argmax over i"""
        neighbor_activations = self.w[maximal_neighbor_activation_indices].squeeze(2) - self.K * distances
        
        neighbor_activation_argmaxes = torch.argmax(neighbor_activations, axis=1, keepdim=True)
        # maximal_neighbor_activation_indices = torch.randint(high=self.k, size=(neighbor_activations.shape[0], 1)).cuda()

        maximal_neighbor_activations = neighbor_activations.gather(1, neighbor_activation_argmaxes)
        # maximal_neighbor_activations = torch.sum(neighbor_activations, dim=1, keepdim=True)
        return maximal_neighbor_activations
    

    def forward(self, X_tilde, bn_importance = 1.0):
        X_tilde = X_tilde.view(X_tilde.shape[0], -1)
        with torch.no_grad():
            _, maximal_neighbor_activation_indices = self.get_approximated_neighbor_activations(X_tilde)

        neighbor_vectors = self.X[maximal_neighbor_activation_indices]  # batchsize x k x img size
        differences = (neighbor_vectors - X_tilde.unsqueeze(1))  # batchsize x k x img size - batchsize x 1 x img size
        distances = torch.norm(differences, dim=2, p=1)

        dists = self.get_maximal_neighbor_activations(distances, maximal_neighbor_activation_indices)
        # return self.bn(dists, bn_importance)
        return dists


    def project_weights(self):
        with torch.no_grad():
            self.w -= self.w.max()



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