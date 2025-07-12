import torch
import torch.nn as nn
import pdb

class DualApproxGD(nn.Module):
    def __init__(self, A, b, initial_guess,iters=10, l1_lmbd=100, dual_dims=(100,100,100), mu_dim=100,device='cpu'):
        """
        A:   [n,n] torch.Tensor (dense or sparse)
        b:   [n]   torch.Tensor
        lr:  step size for the descent
        iters: # of unrolled iters (trade speed vs accuracy)
        l1_lmbd: upper-bound for g1 duals
        """
        super().__init__()
        self.device = device
        self.A = A.to(device)
        self.b = b.to(device)
        self.initial_guess = initial_guess.to(device)
        self.iters = iters
        self.l1_lmbd = l1_lmbd
        self.mu_dim = mu_dim
        self.dual_dim = sum(dual_dims)    # total dim of the SOC dual concatenation
        self.n_mu   = dual_dims[0]   # length of μ (m blocks * block-size)
        self.n_eta  = dual_dims[1]
        self.n_g1   = dual_dims[2]

    def soc_projection(self, v, a):
        # blockwise projection of (v,a) onto {(v,a): ||v||<=a, a>=0}
        norm_v = v.norm(dim=1, keepdim=True)           # assume v is [m, d] and a is [m,1]
        cond1 = (a >= norm_v)
        cond2 = (norm_v <= -a)
        t = 0.5*(a + norm_v)
        v_proj = torch.where(cond1, v,
                     torch.where(cond2, torch.zeros_like(v), t * v / norm_v))
        a_proj = torch.where(cond1, a,
                     torch.where(cond2, torch.zeros_like(a), t))
        return v_proj, a_proj

    def forward(self,lr=1e-4):
        # dual = self.initial_guess   # initial guess
        # dual = torch.ones_like(self.b)   # initial guess
        for it in range(self.iters):
            # gradient of 0.5||A @ dual - b||^2 is Aᵀ (A @dual - b)
            # but since A is square and symmetric in your QP dual, you can do:
            if it == 0:
                grad = (self.A.T).matmul( self.A.matmul(self.initial_guess) - self.b )
            else:
                grad = (self.A.T).matmul( self.A.matmul(dual) - self.b )
            dual = dual - lr * grad

            # now project into the cone:
            # slice out your blocks—here’s pseudocode:
            mu, eta, g1 = dual[:self.n_mu], dual[self.n_mu:self.n_mu+self.n_eta], dual[self.n_mu+self.n_eta:]

            # mu blocks: project each SOC block
            # reshape μ_mu to [num_blocks, block_size]
            # mu = mu.view(self.n_mu//self.mu_dim, self.mu_dim)
            # alphas = mu[:, -1]       # if you stored α as last entry
            # vecs   = mu[:, :-1]
            # v_proj, a_proj = self.soc_projection(vecs, alphas.unsqueeze(dim=1))
            # mu = torch.cat([v_proj,a_proj], dim=1).view(-1).unsqueeze(dim=1)

            mu = torch.clamp(mu, min=0)

            # eta blocks: simple clamp ≥0
            eta = torch.clamp(eta, min=0)

            # g1 blocks: clamp to [0, l1_lmbd]
            # g1 /= self.l1_lmbd
            g1 = torch.clamp(g1, min=0, max=self.l1_lmbd)

            dual_proj = torch.cat([mu, eta, g1], dim=0)

        return dual_proj, grad
