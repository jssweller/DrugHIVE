import torch
import numpy as np
import warnings
from .trainutils import soft_clamp


@torch.jit.script
def jit_sample_normal(mu, sigma):
    ds = mu.mul(0).normal_()
    s = ds.mul_(sigma).add_(mu)
    return s, ds

class NormalDist:
    '''Normal distribution class.'''
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = soft_clamp(mu)
        self.sigma_unscaled = torch.exp(soft_clamp(log_sigma)) + 1e-2
        self.temp = temp  # scaling factor for the variance

        self.sigma = self.sigma_unscaled * self.temp

    @staticmethod
    def init_from_dist_params(dist_params):
        '''Creates new NormalDist from input parameters.'''
        assert dist_params.shape[1] % 2 == 0, 'Invalid input. dist_params must have an even number of channels.'
        n_ch = dist_params.shape[1] // 2

        # dist_params are the means and log_probs
        mu, log_sigma = torch.chunk(dist_params, 2, dim=1)
        return NormalDist(mu, log_sigma)

    def sample(self):
        return jit_sample_normal(self.mu, self.sigma)


    def log_p(self, sample):
        '''Computes the log probability of input sample.'''
        sample0 = (sample - self.mu) / self.sigma # normalized
        vals = - sample0 ** 2 / 2 - np.log(2 * np.pi) / 2 - torch.log(self.sigma)
        return vals


    def kl(self, normal_dist):
        '''Computes kl-divergence between the distribution (self) and another distribution'''
        x1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        x2 = self.sigma / normal_dist.sigma
        return (x1 ** 2 + x2 ** 2) / 2 - torch.log(x2) - 0.5
    
    
    def log_p_sample(self, sample):
        '''Computes log probability of input samples with values on [0,1].'''
        try:
            assert torch.max(sample) <= 1.0 and torch.min(sample) >= 0.0, 'max: %.8f, min: %.8f'%(torch.max(sample), torch.min(sample))
        except Exception as e:
            if torch.max(sample) > 1.02:
                warnings.warn('Warning: Samples outside of range [0,1]. Clipping values. \tmax: %.8f, min: %.8f'%(torch.max(sample), torch.min(sample)))
            sample = torch.clip(sample, max=1.)
            assert torch.max(sample) <= 1.0 and torch.min(sample) >= 0.0, 'max: %.8f, min: %.8f'%(torch.max(sample), torch.min(sample))
        
        sample = 2 * sample - 1.0  # samples assumed to be on [0,1]
        return self.log_p(sample)
    
    
    def samp2vals(self, a):
        '''Converts sample logits to values.'''
        a = (a + 1) / 2
        a = torch.clamp(a, 0, 1)
        return a
    

    def sample_vals(self, t=1.):
        '''Samples from distribution and converts logits to data values.'''
        s, _ = self.sample()
        vals = self.samp2vals(s)
        return vals
    
    
    def sample_vals_mean(self):
        '''Samples from the mean of the distribution.'''
        s = self.mu.clone()
        vals = self.samp2vals(s)
        return vals
    