import unittest
from models.neighbor_discriminator import NeighborDiscriminator
import torch
from torch.autograd import grad


def torch_pairwise_distances(X, Y):
    return torch.sqrt(torch.abs(
        torch.sum(X * X, dim=1, keepdim=True)
        - 2 * torch.matmul(X, Y.transpose(0,1))
        + torch.sum(Y * Y, dim=1, keepdim=True).transpose(0,1)
    ))


class TestNeighborDiscriminator(unittest.TestCase):

    def setUp(self):
        self.X = torch.randn(50000, 3 * 32 * 32)
        self.x_gen = torch.randn(128, 3 * 32 * 32)
        self.x_gen.requires_grad = True
        self.dis = NeighborDiscriminator(self.X, k=20, K=1)
        self.dis.w.data = torch.randn_like(self.dis.w.data) / 10
        self.dis.update_index()


    def test_neighbor_activation_correctness(self):
        """For the baseline (exact) NeighborDiscriminator, we should have exact matches"""
        maximal_neighbor_activations = self.dis(self.x_gen)[0]

        pairwise_distances = torch_pairwise_distances(self.X, self.x_gen)
        exact_neighbor_activations = -self.dis.K * pairwise_distances + self.dis.w.data
        exact_maximal_neighbor_activations, indices = torch.max(exact_neighbor_activations, axis=0)

        self.assertEqual(maximal_neighbor_activations.shape, exact_maximal_neighbor_activations.shape)
        errs = torch.abs(maximal_neighbor_activations - exact_maximal_neighbor_activations)

        # This can be a little bigger because torch pairwise distance calcs are notoriously unstable
        self.assertTrue(all(errs < 1e-3))

    def test_input_gradient_correctness(self):
        """The gradient for each \tilde x_j \in x_gen is of the form

            -K (\tilde x_j - x_i) / (\|x_j - x_i)

            Where i is the index of the maximal neighbor activation.
        """
        # Get the gradient from the model
        d_gen = self.dis(self.x_gen)[0]
        loss_gen = torch.sum(d_gen)
        grad_model = grad(outputs=loss_gen, inputs=self.x_gen)

        # Get the exact gradient of the maximal neighbor activation w.r.t the input
        pairwise_distances = torch_pairwise_distances(self.X, self.x_gen)
        exact_neighbor_activations = -self.dis.K * pairwise_distances + self.dis.w.data
        exact_maximal_neighbor_activations, _ = torch.max(exact_neighbor_activations, axis=0)
        loss_gen_exact = torch.sum(exact_maximal_neighbor_activations)
        grad_exact = grad(outputs=loss_gen_exact, inputs=self.x_gen)

        errs = torch.abs(grad_exact[0] - grad_model[0]).view(-1)
        self.assertTrue(all(errs < 1e-5))



