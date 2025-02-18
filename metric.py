import numpy as np
import torch 

class metric:

    def __init__(self, g, beta, alp):
        assert g.shape[0] == g.shape[1] == 3, "Metric tensor must be square"
        self.sqrtg = torch.sqrt(torch.linalg.det(g))
        self.invg  = torch.linalg.inv(g)
        self.g     = g
        self.beta  = beta
        self.alp   = alp

    def raise_index(self,covec):
        """Raise the index of a single covariant vector or a batch."""
        if covec.dim() == 1 :
            return torch.matmul(self.invg, covec)
        else:
            return torch.matmul(self.invg, covec.T).T

    def lower_index(self,vec):
        """Lower the index of a single contravariant vector or a batch."""
        if vec.dim() == 1 :
            return torch.matmul(self.g, vec)
        else:
            return torch.matmul(self.g, vec.T).T

    def square_norm_upper(self,vec):
        """Compute square norm of a single contravariant vector or a batch."""
        if vec.dim() == 1:
            return torch.matmul(vec, torch.matmul(self.g, vec))
        else:    
            return torch.einsum('bi,ij,bj->b', vec, self.g, vec)

    def square_norm_lower(self,covec):
        """Compute square norm of a single covariant vector or a batch."""
        if covec.dim() == 1:
            return torch.matmul(covec, torch.matmul(self.invg, covec))
        else:
            return torch.einsum('bi,ij,bj->b', covec, self.invg, covec)


    
