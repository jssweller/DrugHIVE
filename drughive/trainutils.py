from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import yaml

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.jit.script
def soft_clamp(x: torch.Tensor, min: float = -5, max: float = 5):
    d = (max-min)/2
    return (x-(d+min)).div(d).tanh_().mul(d) + (d + min)


def average_distributed(x):
    '''Averages a distributed tensor.'''
    dist.all_reduce(x.data, op=dist.ReduceOp.SUM)
    x.data = x.data / float(dist.get_world_size())


class KLBalancer(object):
    def __init__(self,  groups_per_res) -> None:
        self. groups_per_res =  groups_per_res
        self.kl_coeff = 1
        self.init_alpha_i(self. groups_per_res)
    

    def init_alpha_i(self,  groups_per_res=None):
        g = groups_per_res
        if g is None:
            g = self.groups_per_res

        self.alpha_i = torch.cat([2**(2*i) / g[-i-1] * torch.ones(g[-i-1]) for i in range(len(g))], dim=0)
        self.alpha_i /= torch.min(self.alpha_i) # set min(coeff) = 1

     
    def balance(self, kl_all):
        alpha_i = self.alpha_i
        kl_coeff = self.kl_coeff
        kl_all = torch.stack(kl_all, dim=1)
        kl_out = kl_all
        kl_vals = torch.mean(kl_all, dim=0)

        if kl_coeff < 1.0:
            if alpha_i.ndim == 1:
                alpha_i = alpha_i.unsqueeze(0)

            kl_i = torch.mean(torch.abs(kl_all), dim=0, keepdim=True) + 0.01 # kl per group
            kl_sum = torch.sum(kl_i)

            kl_i = kl_i / alpha_i * kl_sum
            kl_i = kl_i / torch.mean(kl_i, dim=1, keepdim=True)
            kl_out = kl_all * kl_i.detach()

            kl_coeffs = kl_i.squeeze(0)
        else:
            kl_coeffs = torch.ones(size=(kl_all.shape[1],))
        
        kl_out = kl_coeff * torch.sum(kl_out, dim=1)
        return kl_out, kl_coeffs, kl_vals


    def update_kl_coeff(self, step, steps_anneal, steps_constant, min_val):
        x = min((step - steps_constant) / steps_anneal, 1.0)
        self.kl_coeff = max(x, min_val)


    def to_device(self, device):
        self.device = device
        self.alpha_i = self.alpha_i.to(device)


def recon_loss_fn(dist, x, zero_alpha=.1, reduce=True):
    if zero_alpha != 1:
        xmask = (x==0) # mask empty spaces in grid
    recon = dist.log_p_sample(x)

    if zero_alpha != 1: # scale loss values of empty grid locations by a factor of zero_alpha
        xmask = xmask.all(dim=1).unsqueeze(1)
        recon *= (~xmask + zero_alpha * xmask)

    if reduce:
        recon = torch.sum(recon, dim=[1, 2, 3, 4])

    return - recon


def steric_loss_fn(dist, x, mean_only=False):
    x = torch.amax(x, dim=1, keepdim=True)

    if mean_only:
        # minimize overlap (element-wise product) of predicted mean and protein density
        mu = dist.mu
        mu = torch.clamp(mu, -1, 1.)/2 + 0.5
        steric = x * mu
    else:
        # minimize squared probability
        steric = torch.exp(dist.log_p_sample(x)) ** 2
        steric *= (x > 0.)
        steric = soft_clamp(steric, min=-5, max=5)

    return torch.sum(steric, dim=[1, 2, 3, 4])


def spec_norm_loss_fn(model):
    # call after forward pass
    u = model.spectral_norm_u
    v = model.spectral_norm_v

    weights = defaultdict(list)   # weights[weight.shape] = reshaped weight
    for l in model.layers_conv_all:
        weight = l.weight_n # normalized weight
        weight_flat = weight.reshape(weight.size(0), -1)
        weights[weight_flat.shape].append(weight_flat)

    loss = 0
    for shape in weights:
        weights[shape] = torch.stack(weights[shape], dim=0)
        with torch.no_grad():
            n_power_iterations = 4
            if shape not in u:
                n_power_iterations = 10 * n_power_iterations # increase first time
                num_w, row, col = weights[shape].shape
                u[shape] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                v[shape] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)

            for j in range(n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v[shape] = F.normalize(torch.matmul(u[shape].unsqueeze(1), weights[shape]).squeeze(1), dim=1, eps=1e-5)
                u[shape] = F.normalize(torch.matmul(weights[shape], v[shape].unsqueeze(2)).squeeze(2), dim=1, eps=1e-5)

        sigma = torch.matmul(u[shape].unsqueeze(1), torch.matmul(weights[shape], v[shape].unsqueeze(2)))
        loss += torch.sum(sigma)
    
    return loss


def batchnorm_loss_fn(model):
    loss_bn = 0
    for layer in model.layers_bn_all:
        if layer.affine:
            loss_bn += torch.max(torch.abs(layer.weight))
    return loss_bn


def shuffle_channels(x):
    for i in range(len(x)):
        x[i] = x[i, torch.randperm(x.shape[1])]
    return x

def get_gradients_max(model):
    '''Calculates the max of the model gradients.'''
    parameters = [p for p in model.parameters() if p is not None and p.requires_grad]
    return torch.stack([p.grad.detach().abs().max() for p in parameters]).max().cpu()


def get_gradients_norm(model, norm_type=2):
    '''Calculates the norm of the model gradients.'''
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), 2.0).item()
    return total_norm    

def filter_predictions(true_grids, pred_grids):
    '''Gets values from true and predicted tensor grids where both grids are nonzero.'''
    assert len(pred_grids) == len(true_grids), 'grid list lengths must match'

    nch = pred_grids.shape[1]
    true_vals = []
    pred_vals = []
    
    for k, (tgrid, pgrid) in enumerate(zip(true_grids, pred_grids)):
        tz = tgrid != 0
        pz = pgrid != 0

        tpz = torch.logical_or(tz,pz)

        for i in range(nch):                
            if k == 0:
                true_vals.append([])
                pred_vals.append([])
            
            if tpz.sum() == 0:
                true_vals[i].append(torch.zeros(1))
                pred_vals[i].append(torch.zeros(1))
            else:
                true_vals[i].append(tgrid[i][tpz[i]])
                pred_vals[i].append(pgrid[i][tpz[i]])

    for i in range(nch):
        true_vals[i] = np.concatenate(true_vals[i])
        pred_vals[i] = np.concatenate(pred_vals[i])
    return true_vals, pred_vals


def plot_scatterplot(true_vals, pred_vals, channels, colors, alpha=1):
    '''Plots pearsonr scatterplot given true and predicted values.'''
    nch = len(true_vals)
    
    ncols = min(4, nch)
    nrows = max(1, - (nch // -ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if ncols == 1 and nrows == 1:
        axs = np.array([axs])
    if nrows == 1:
        axs = axs.reshape(1,-1)


    for i in range(len(axs.flatten())):
        ax = axs[i//ncols, i%ncols]
        if i > nch-1:
            ax.set_visible(False)
            continue

        label = str(channels[i])
        c = colors[i]
        if 'rgb' in c:
            c = np.array(c.replace('rgb','')[1:-1].split(','), dtype=int)/255

        try:
            pr, pval = pearsonr(true_vals[i], pred_vals[i])
        except:
            pr = float('nan')
            pass

        x = true_vals[i]
        y = pred_vals[i]

        ax.scatter(x, y, s=5, color=c, alpha=alpha)
        ymax = max(x.max(), y.max())
        ymax = min(max(ymax,0.1),1)
        ax.plot([0,ymax], [0,ymax], color='grey')
        ax.set_xlabel('true')
        ax.set_ylabel('pred')
        ax.set_title(label + '\nR=%.2f'%pr)
    fig.tight_layout()

    return fig, axs


def plot_confusion_matrix(conf_mat, names, fig=None, ax=None, colorbar=False):
    '''Plots confusion matrix given matrix values.'''
    if fig is None:
        fig = plt.figure(figsize=(10,8))
    if ax is None:
        ax = plt.subplot(111)
    nch = len(conf_mat)
    im = ax.imshow(conf_mat, vmin=0, vmax=1)
    if '[' in str(names[0]):
        ax.set_yticks(np.arange(nch), [str(x)[1:-1].replace('\'','').replace(',','\n') for x in names], size=12)
        ax.set_xticks(np.arange(nch), [str(x)[1:-1].replace('\'','') for x in names], size=12)
    else:
        ax.set_yticks(np.arange(nch), names, size=12)
        ax.set_xticks(np.arange(nch), names, size=12)
    ax.xaxis.tick_top()
    for i in range(nch):
        for j in range(nch):
            c = 'w' if conf_mat[i,j] < 0.7 else 'k'
            ax.text(j,i, s='%d'%(conf_mat[i,j]*100), ha='center', va='center', color=c, size=9)

    if colorbar:
        fig.colorbar(im, ax=ax)
    return fig, ax


def get_confusion_matrix(true_grids, pred_grids):
    '''Gets confusion matrix given true and predicted grids.'''
    assert len(pred_grids) == len(true_grids), 'grid list lengths must match'

    nch = pred_grids.shape[1]
    N = len(pred_grids)
    conf = np.zeros((N, nch,nch))
    mcounts = np.zeros((N, nch))

    # thresh = 3e-5
    thresh = 1e-2

    for k, (mgrid, egrid) in enumerate(zip(true_grids, pred_grids)):
        for i in range(nch):
            mvals = mgrid[i]
            counts = (mvals > thresh).sum()
            mcounts[k,i] = counts
            for j in range(nch):
                evals = egrid[j]
                if i != j:
                    evals = evals * (mgrid[j] == 0)  # zero any values where there may be overlap between channels
                conf[k,i,j] = torch.logical_and(evals > thresh, mvals > thresh).sum()

    return conf.sum(axis=0)/(mcounts.sum(axis=0).reshape(-1,1) + 1)
    

class Hparams(object):
    '''Class for storing parameters.'''
    def __init__(self, recursive=True, level=0, **kwargs) -> None:
        self._hparams_level = level
        self._hparams_recursive = recursive
        if recursive:
            for k,v in kwargs.items():
                if isinstance(v,dict):
                    v = Hparams(**v, recursive=True, level=self._hparams_level+1)
                self.__dict__[k] = v
        else:
            self.__dict__.update(kwargs)


    def __len__(self):
        return len([k for k in self.__dict__.keys() if '_hparams' not in k])

    def __repr__(self):
        indent = '    '*self._hparams_level
        s = ''
        for k,v in self.__dict__.items():
            if '_hparams' in k:
                continue
            s += '\n'+indent
            if isinstance(v, dict) or isinstance(v, Hparams):
                if len(v) < 3:
                    if isinstance(v, Hparams):
                        v = v.dict
                    s += k + ': ' + str(v)
                    continue
            s += k + ': ' + v.__repr__()
        return s

    def __iter__(self):
        return (x for x in self.dict.items())
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        setattr(self, key, val)
    

    @property
    def dict(self):
        d = {}
        for k,v in self.__dict__.items():
            if '_hparams' in k:
                continue
            if isinstance(v,Hparams):
                v = v.dict
            d[k] = v
        return d
    
    def get(self, key, default_val=None):
        return self.dict.get(key, default_val)

    def load_yaml(self, file):
        with open(file, 'r') as f:
            args = yaml.safe_load(f)
        self.__init__(recursive=self._hparams_recursive, **args)

    def save_yaml(self, file):
        with open(file, 'w+') as f:
            yaml.dump(self.dict, f, default_flow_style=None, indent=4, sort_keys=False)