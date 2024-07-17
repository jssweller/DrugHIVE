
import os, sys
from copy import copy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import itertools

import torch
from torch import nn

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from openbabel import openbabel as ob
from openbabel import pybel

from .gridutils import *
from .molecules import AtomTyperDefault, BaseAtomTyper, Mol2, get_largest_fragment, update_coordinates, rotate_coordinates, translate_coordinates
from .atomgridding import GridEncoder
from .blocks import GaussConvLayer



def lj_potential(r, rmin=1.3, p1=12, p2=6, eps=6):
    '''Lennard-Jones potential'''
    sigma = rmin*(p1/p2)**(1/(p2-p1))
    return eps * ((sigma/r)**(p1) - (sigma/r)**(p2))

def prune_coords(coords, dist_thresh=0.9, coords_fix=[]):
    '''Prunes coordinates within distance threshold of other atoms. Pruning priority is in reversed order of supplied coords (last coords pruned first).'''
    dmat = squareform(pdist(coords.reshape(-1,3)))
    dmat += np.eye(len(dmat))*100
    coords_prune = []
    for i in range(len(dmat)):
        for j in reversed(range(i+1,len(dmat))):
            if (j in coords_prune) or (j in coords_fix):
                continue
            if dmat[i,j] < dist_thresh:
                coords_prune.append(j)
                dmat[j,:] = 100
                dmat[:, j] = 100
    idxs_keep = np.array([i not in coords_prune for i in range(len(coords))], dtype=bool)
    return coords[idxs_keep], idxs_keep

def set_aromaticity(mol):
    '''Sets aromaticity of a molecule.'''
    mol = Chem.RWMol(mol)
    
    rinfo = mol.GetRingInfo()
    for ring, bond_ring in zip(rinfo.AtomRings(), rinfo.BondRings()):
        if all([int(mol.GetAtomWithIdx(i).GetAtomicNum()) in [6, 7] for i in ring]) and len(ring) in [4,5,6]:
            atom_aroms = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring]
            bond_aroms = [mol.GetBondWithIdx(i).GetIsAromatic() for i in bond_ring]
            bond_types = [mol.GetBondWithIdx(i).GetBondType() for i in bond_ring]
            for i in ring:
                mol.GetAtomWithIdx(i).SetIsAromatic(True)
            for i in bond_ring:
                mol.GetBondWithIdx(i).SetIsAromatic(True)
                mol.GetBondWithIdx(i).SetBondType(Chem.BondType.AROMATIC)
                
            problems = Chem.DetectChemistryProblems(mol)
            if len(problems) > 0:
                if not (len(problems) == 1 and problems[0].GetType() in ['AtomValenceException']):
                    for i, ai in enumerate(ring):
                        mol.GetAtomWithIdx(ai).SetIsAromatic(atom_aroms[i])
                    for i, bi in enumerate(bond_ring):
                        mol.GetBondWithIdx(bi).SetIsAromatic(bond_aroms[i])
                        mol.GetBondWithIdx(bi).SetBondType(bond_types[i])
    return Chem.Mol(mol)


def clean_ring_aromaticity(mol):
    '''Sets atoms and bonds that are not in aromatic rings as non-aromatic.'''
    mol = Chem.RWMol(mol)
    rinfo = mol.GetRingInfo()
    
    # get list of atoms in aromatic rings
    ar_atoms = []
    ar_bonds = []
    for ring, bond_ring in zip(rinfo.AtomRings(), rinfo.BondRings()):
        atom_aroms = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring]
        bond_aroms = [mol.GetBondWithIdx(i).GetIsAromatic() for i in bond_ring]
        if all(atom_aroms) and all(bond_aroms):
            ar_atoms.extend(ring)
            ar_bonds.extend(bond_ring)
        
    
    for ring, bond_ring in zip(rinfo.AtomRings(), rinfo.BondRings()):
        atom_aroms = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring]
        bond_aroms = [mol.GetBondWithIdx(i).GetIsAromatic() for i in bond_ring]
        if not (all(atom_aroms) and all(bond_aroms)):            
            for i in ring:
                if i not in ar_atoms:
                    mol.GetAtomWithIdx(i).SetIsAromatic(False)
            for i in bond_ring:
                if i not in ar_bonds:
                    mol.GetBondWithIdx(i).SetIsAromatic(False)
                    mol.GetBondWithIdx(i).SetBondType(Chem.BondType.SINGLE)
    return Chem.Mol(mol)


def clean_nonring_aromaticity(mol):
    '''Sets atoms and bonds that are not in rings as non-aromatic.'''
    for atom in mol.GetAtoms():
        if not atom.IsInRing():
            atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            bond.SetIsAromatic(False)
            if bond.GetBondType == Chem.BondType.AROMATIC:
                bond.SetBondType(Chem.BondType.SINGLE)


def obmol2rdmol(mol):
    '''Converts openbabel molecule to an rdkit molecule.'''
    rdmol = Chem.RWMol()
    conf = Chem.Conformer(mol.NumAtoms())
    
    for i, atom in enumerate(ob.OBMolAtomIter(mol)):
        rdatom = Chem.Atom(atom.GetAtomicNum())
        rdatom.SetIsAromatic(atom.IsAromatic())
        rdatom.SetNumExplicitHs(atom.GetImplicitHCount())
        
        rdmol.AddAtom(rdatom)
        coords = (atom.GetX(), atom.GetY(), atom.GetZ())
        conf.SetAtomPosition(i, Point3D(*list(coords)))    
    
    rdmol.AddConformer(conf)
    bo_dict = {1: Chem.BondType.SINGLE,
              2: Chem.BondType.DOUBLE,
              3: Chem.BondType.TRIPLE
             }
    
    for bond in ob.OBMolBondIter(mol):
        i = bond.GetBeginAtomIdx() - 1
        j = bond.GetEndAtomIdx() - 1
        
        
        bond_order = bo_dict.get(bond.GetBondOrder(), None)
        if bond_order is None:
            raise Exception('Invalid bond order: %s'%str(bond.GetBondOrder()))
        
        rdmol.AddBond(i,j, bond_order)
        rdmol.GetBondBetweenAtoms(i,j).SetIsAromatic(bond.IsAromatic())
    
    Chem.GetSSSR(rdmol)
    return Chem.Mol(rdmol)

        
def rdmol2obmol(rdmol):
    '''Converts rdkit molecule to an openbabel molecule.'''
    obmol = ob.OBMol()
    obmol.BeginModify()
    
    conf = rdmol.GetConformer(0)
    positions = conf.GetPositions()
    for i, atom in enumerate(rdmol.GetAtoms()):
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(atom.GetAtomicNum())
        obatom.SetFormalCharge(atom.GetFormalCharge())
        obatom.SetAromatic(atom.GetIsAromatic())
        obatom.SetImplicitHCount(atom.GetNumExplicitHs())
        obatom.SetVector(*list(positions[i]))
        
    bo_dict = {Chem.BondType.SINGLE: 1,
               Chem.BondType.DOUBLE: 2,
               Chem.BondType.TRIPLE: 3,
               Chem.BondType.AROMATIC: 1,
             }
    
    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx() + 1
        j = bond.GetEndAtomIdx() + 1
        
        bond_order = bo_dict.get(bond.GetBondType(), None)
        if bond_order is None:
            raise Exception('Invalid bond order: %s'%str(bond.GetBondType()))      
        obmol.AddBond(i, j, bond_order)   

    obmol.EndModify()
    return obmol

def set_aromaticity_ob(obmol):
    obmol.SetAromaticPerceived(False)
    for i, atom in enumerate(ob.OBMolAtomIter(obmol)):
        atom.IsAromatic()
    
    for bond in ob.OBMolBondIter(obmol):
        a0 = bond.GetBeginAtom()
        a1 = bond.GetEndAtom()
        bond.SetAromatic(bond.IsInRing() and a0.IsAromatic() and a1.IsAromatic())
    obmol.SetAromaticPerceived(True)
    

def perceive_hybridization_ob(obmol):
    obmol.SetHybridizationPerceived(False)
    for atom in ob.OBMolAtomIter():
        atom.GetHyb()
    obmol.SetHybridizationPerceived(True)
    

def perceive_bond_orders_ob(obmol):
    obmol.SetHydrogensAdded(False)
    obmol.AddHydrogens()
    for a in ob.OBMolAtomIter(obmol):
            if a.GetAtomicNum() == 1:
                a.SetHyb(1)
    obmol.PerceiveBondOrders()
    obmol.SetHydrogensAdded(True)


class AtomGenerator(GridEncoder):
    def __init__(self, atom_size, conv_gauss, resolution: float = 1, values: torch.Tensor = None, channels: list = ['default'], center: tuple = (0,0,0), n_batch: int = 1, device: str = 'cpu', atomtyper: BaseAtomTyper = None):
        '''Class for generating atomic densities on a grid.'''
        grid_size = (atom_size + conv_gauss.weight.shape[-1]//2)
        super().__init__(grid_size, resolution, values, channels, center, n_batch, device, atomtyper)
        self.atom_size = atom_size
        self.atom_coords = torch.zeros((1,1,3)) - self.resolution/2
        self.atom_types = self.atomtyper.types2vec(['default']).unsqueeze(0)
        self.conv_gauss = GaussConvLayer(conv_gauss.var, conv_gauss.trunc, conv_gauss.resolution, n_channels=1)
        self.init_grid()

    @property
    def atom(self):
        d = (self.grid_size - self.atom_size) // 2
        return self.values[0, 0, d:-d-1, d:-d-1, d:-d-1]

    def generate_atom(self, dr=torch.zeros(3)):
        if isinstance(dr,float):
            dr = torch.ones(3) * dr
        if not isinstance(dr, torch.Tensor):
            dr = torch.tensor(dr, dtype=float)

        assert dr.abs().max() <= 0.5, 'Values of `dr` must be in [-0.5, 0.5]. Otherwise you are in neighboring voxel! Received %s'%(str(dr))

        self.atom_coords = 0*self.atom_coords - self.resolution/2 + dr
        self.encode_coords2grid()
        self.values = self.conv_gauss(self.values.to(self.conv_gauss.weight.device))
        return self.atom
        

class DensityFitter(object):
    def __init__(self, conv_gauss, device='cpu', atom_size=None, conv_kernel_size=None) -> None:
        '''Class for fitting atomic coordinates to a density grid.'''
        self.device = device
        self.conv_gauss = conv_gauss
        self.n_channels = conv_gauss.in_channels
        self.atom_size = atom_size
        self.conv_kernel_size = conv_kernel_size

        self.init_conv_filter(self.conv_kernel_size)

        self.init_atom_generator(atom_size)
        self.atom0 = self.atom_gen.generate_atom(dr=0).clone()

    
    def init_conv_filter(self, kernel_size=None):
        '''Initializes convolution layer that filters raw density for better peak picking.'''
        conv_kernel = self.conv_gauss.weight.clone().cpu()

        if kernel_size is None or kernel_size >= self.conv_gauss.kernel_size[0]:
            self.conv_filter = self.conv_gauss
        else:
            assert kernel_size % 2 == 1, 'kernel_size must be odd, received %d'%kernel_size
            
            self.conv_filter = nn.Conv3d(in_channels=self.n_channels, out_channels=self.n_channels, groups=self.n_channels, kernel_size=kernel_size, padding=1, bias=False)    
            
            d = (self.conv_gauss.kernel_size[0] - kernel_size) // 2
            self.conv_filter.weight.data = conv_kernel[:,:,d:-d,d:-d,d:-d]
            self.conv_filter.to(self.device)

    @property
    def atom_size(self):
        return self._atom_size

    @atom_size.setter
    def atom_size(self, value):
        if value is not None:
            self._atom_size = value
            self.init_atom_generator(value)

    def init_atom_generator(self, atom_size=None):
        if atom_size is None:
            atom_size = self.conv_gauss.weight.shape[-1]
        self.atom_gen = AtomGenerator(conv_gauss=self.conv_gauss, atom_size=7, resolution=1, device=self.device)

    def to(self, device):
        self.device = device
        self.conv_filter.to(device)
        self.atom_gen.to(device)
        self.atom0.to(device)

    def get_pred_grid(self):
        self.pred_grid = self.target_grid.copy()
        self.pred_grid.atom_coords = self.coords_pred.cpu()
        self.pred_grid.atom_types = self.types_pred_1hot.cpu()
        self.pred_grid.values = self.pred_grid.values.cpu()

        self.pred_grid.encode_coords2grid()
        self.pred_grid.values = self.conv_gauss(self.pred_grid.values.to(self.conv_gauss.weight.device)).cpu()
        return self.pred_grid

    def fit_density(self, 
                    target_grid, 
                    coords_start=None, 
                    types_start=None,
                    boxes_mod=None,
                    density_start=None, 
                    fit_kernel_size=1, 
                    amp_thresh=0.3, 
                    dist_thresh=0.9,
                    device='cpu', 
                    coords_fix=[], 
                    verbose=False):
        
        self.fit_kernel_size = fit_kernel_size
        self.verbose = verbose

        self.boxes_mod = boxes_mod
        self.target_grid = target_grid
        atom = self.atom0
        self.conv_filter.to(device)


        self.grid_vals = target_grid.values.clone().float().to(device)
        self.grid_vals0 = self.grid_vals.clone() # used for weighted coordinate calculation

        if density_start is not None:
            self.grid_vals -= density_start.to(device)

        self.coords_grid_pred = []
        self.types_pred = []
        if coords_start is not None:
            self.coords_grid_pred = coords_start
        if types_start is not None:
               self.types_pred = types_start
        

        self.dist_thresh = dist_thresh/target_grid.resolution
        
        self.amp_thresh = amp_thresh
        self.grid_vals_filt = self.conv_filter(self.grid_vals)

        self.idxs_flat_filt = torch.stack(torch.ones_like(self.grid_vals_filt).nonzero(as_tuple=True), dim=1) # indices of flattened grid_vals_filt tensor
        self.idxs_flat = torch.stack(torch.ones_like(self.grid_vals).nonzero(as_tuple=True), dim=1) # indices of flattened grid_vals tensor
        self.idxs_3d = torch.stack(torch.meshgrid(*[torch.arange(self.grid_vals.shape[-1])]*3, indexing='ij'), dim=-1) # 3d grid indices

        self.i = 0

        self.grid_vals_mod = self.grid_vals
        while self.grid_vals_mod.max() > self.amp_thresh and self.i < 100:
            self.fit_next_atom()

        if len(self.coords_grid_pred) == 0:
            return self.coords_grid_pred, torch.empty(1)
            
        self.coords_grid_pred = [torch.cat(x, dim=0).mean(dim=0).reshape(-1,3) if i not in coords_fix else x[0] for i,x in enumerate(self.coords_grid_pred)] # take average of coordinates for each channel
        self.coords_grid_pred = torch.cat(self.coords_grid_pred).reshape(-1,3)
        
        
        self.types_pred = [x[:1] for x in self.types_pred] # choose first type (highest amplitude)


        
        try:
            self.types_pred_1hot = torch.cat([target_grid.atomtyper.nums2vec(np.unique(np.array([target_grid.channel_nums[t][0] for t in tlist]).flatten())).reshape(1,-1) for tlist in self.types_pred]).unsqueeze(0)
        except Exception as e:
            print('\nERROR:')
            print('\ninner:')
            print(np.unique(np.array([target_grid.channel_nums[t][0] for t in self.types_pred[0]]).flatten()))
            print('\ntypes_pred:', self.types_pred)
            print('target_grid channels:', target_grid.channel_nums)
            print('\n')
            raise e

        self.coords_pred = target_grid.voxels2angstroms(self.coords_grid_pred)
        return self.coords_pred, self.types_pred_1hot
    
    
    def calc_weighted_coord(self, grid_vals, c, t, kernel_size=3):
        cr = c.round().long().squeeze()

        k = kernel_size
        d = k//2
        a = grid_vals[0,t,cr[0]-d:cr[0]+d+1, cr[1]-d:cr[1]+d+1, cr[2]-d:cr[2]+d+1].clone()
        a /= a.sum()

        c = torch.stack(torch.meshgrid(*[torch.arange(k)-k//2]*3, indexing='ij'), axis=-1) # indices of subgrid
        try:
            w = (a.unsqueeze(-1) * c).reshape(-1,3)
        except Exception as e:
            print('a', a.shape, '\t', a)
            print('c', c.shape, '\t', cr)
            raise e
  
        dr = w.sum(dim=0)
        return dr

    def fit_next_atom(self, farthest_point=True):
        rmax = torch.zeros(5)
        while (rmax[2:].min() <= 1) or (rmax[2:].max() >= self.grid_vals.shape[-1] - 2):
            imax = self.grid_vals_filt.argmax()
            rmax = self.idxs_flat[imax]
            self.grid_vals_filt.view(-1)[imax] = 0

        atom = self.atom0
        
        x = rmax[-3]
        y = rmax[-2]
        z = rmax[-1]
        
        t = int(rmax[-4])
        cr = torch.tensor([[x,y,z]]).float()

        if self.fit_kernel_size > 1:
            dr = self.calc_weighted_coord(self.grid_vals0, cr, t, kernel_size=self.fit_kernel_size)
            dr = torch.clip(dr, min=-0.5, max=0.5)
            atom = self.atom_gen.generate_atom(dr=dr)
        else:
            dr = 0
        c = cr + dr

        cdists = np.zeros(1)

        coords_include = []
        if len(self.coords_grid_pred) > 0:
            for xx in self.coords_grid_pred:
                if isinstance(xx, list):
                    # print('xx', torch.cat(xx).shape)
                    coords_include.append(torch.cat(xx).mean(dim=0).reshape(1,3))
                else:
                    coords_include.append(xx)
                
        if len(self.coords_grid_pred) > 0 and len(coords_include) > 0:
            cdists = torch.norm(c - torch.cat(coords_include), dim=-1)
            

        if cdists.min() > self.dist_thresh or len(self.coords_grid_pred) == 0 or len(coords_include) == 0:
            self.coords_grid_pred.append([c])
            self.types_pred.append([t])
        else:
            idx = cdists.argmin()
            self.types_pred[idx].append(t)
            self.coords_grid_pred[idx].append(c)
        
        # subtract current atom
        x0,x1 = x-3, x+4
        y0,y1 = y-3, y+4
        z0,z1 = z-3, z+4

        gshape = self.grid_vals.shape

        dx0,dx1 = abs(min(x0, 0)), len(atom) - max(0, (x1-gshape[2]))
        dy0,dy1 = abs(min(y0, 0)), len(atom) - max(0, (y1-gshape[3]))
        dz0,dz1 = abs(min(z0, 0)), len(atom) - max(0, (z1-gshape[4]))

        x0, x1 = max(x0,0), min(x1, gshape[2])
        y0, y1 = max(y0,0), min(y1, gshape[3])
        z0, z1 = max(z0,0), min(z1, gshape[4])


        self.grid_vals[rmax[0],:, x0:x1, y0:y1, z0:z1] =  torch.clip(self.grid_vals[rmax[0],:, x0:x1, y0:y1, z0:z1] - atom[dx0:dx1, dy0:dy1, dz0:dz1], min=0)
        self.grid_vals_filt = self.conv_filter(self.grid_vals)

        self.grid_vals_mod = self.grid_vals
        if self.boxes_mod is not None:
            gvals = torch.zeros_like(self.grid_vals_filt)
            gvals_mod = torch.zeros_like(self.grid_vals)
            for b in self.boxes_mod:
                gvals[:, :, b[0,0]:b[1,0], b[0,1]:b[1,1], b[0,2]:b[1,2]] = self.grid_vals_filt[:, :, b[0,0]:b[1,0], b[0,1]:b[1,1], b[0,2]:b[1,2]]
                gvals_mod[:, :, b[0,0]:b[1,0], b[0,1]:b[1,1], b[0,2]:b[1,2]] = self.grid_vals[:, :, b[0,0]:b[1,0], b[0,1]:b[1,1], b[0,2]:b[1,2]]
            self.grid_vals_filt = gvals
            self.grid_vals_mod = gvals_mod
        self.i += 1


    def optimize_coordinates(self, 
                             grid_pred, 
                             true_grid_vals,
                             max_iter=20,
                             lr=5e-2,
                             early_stop=False, 
                             early_stopping_warmup=2, 
                             early_stop_trough_num=1,
                             coords_true=None,
                             opt_steric=False,
                             opt_bond_dist=False,
                             opt_bond_w=1.,
                             steric_dist=0.9,
                             opt_steric_w=1.,
                             fix_coords=[],
                             debug=False, 
                             verbose=False):

        if torch.cuda.is_available():
            self.conv_gauss = self.conv_gauss.cuda()
                             
        self.opt_grid = grid_pred.copy()
        coords0 = grid_pred.atom_coords.clone()
        coords_start = coords0.clone()
        coords0.requires_grad = True
        
        self.true_grid_vals = true_grid_vals
        self.true_grid_vals = self.true_grid_vals.cuda()

        optimizer = torch.optim.Adam((coords0,), lr=lr, betas=(0.8,0.999))
        
        # self.loss_func = nn.functional.l1_loss
        self.loss_func = nn.SmoothL1Loss(beta=0.03, reduction='none')


        self.losses = []

        if debug:
            assert coords_true is not None, 'If debug == True, must provide coords_true.'
            self.coords_true = coords_true
            self.mean_dist = []
            self.max_dist = []
            self.coords_list = []

        loss_troughs = 0
        loss_iter = 0
        for i in range(max_iter):
            optimizer.zero_grad()
            self.opt_grid.values = self.opt_grid.values.detach() # breaks computation graph between iterations    

            self.opt_grid.atom_coords = coords0
            self.opt_grid.encode_coords2grid()

            self.pred_grid_vals = self.conv_gauss(self.opt_grid.values.to(self.conv_gauss.weight.device))
            
            loss = self.loss_func(self.pred_grid_vals, self.true_grid_vals)
#             nz_mask = self.true_grid_vals == 0
#             loss[nz_mask] = loss[nz_mask] * 0.5
            loss = loss.sum()
            
            if opt_steric:
                dists = torch.cdist(coords0, coords0) + 100*torch.eye(coords0.shape[1])
                steric_loss = (torch.clip((steric_dist - dists), min=0)**2).sum()
                loss += opt_steric_w * steric_loss

            if opt_bond_dist:
                dists = torch.cdist(coords0, coords0) + 10*torch.eye(coords0.shape[1])
                dist_loss = lj_potential(dists, rmin=1.3, p1=3.1, p2=3.).sum()/2
                loss += dist_loss * opt_bond_w
            
            self.losses.append(loss.detach().cpu().numpy())
            loss_iter += 1


            if debug and coords_true is not None:
                coords_pred = self.opt_grid.atom_coords[0].clone().detach().cpu()
                if i==0:
                    atom_pairs, _, _ = align_coords(coords_pred, coords_true)
                    atom_pairs = atom_pairs[np.argsort(atom_pairs[:,1])]

                coords_pred = coords_pred[atom_pairs[:,0]]
                coords_true = coords_true[atom_pairs[:,1]]

                dist_norm = np.linalg.norm(coords_pred - coords_true, axis=-1)
                self.max_dist.append(dist_norm.max())
                self.mean_dist.append(dist_norm.mean())
                self.coords_list.append(coords_pred)

            if early_stop and loss_iter > max(early_stopping_warmup, 0):
                if self.losses[-1] > self.losses[-2]:
                    loss_troughs += 1
                    loss_iter = 0
                    if loss_troughs == early_stop_trough_num:
                        optimizer.zero_grad()
                        break

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for j in fix_coords:
                    coords0[0,j,:] = coords_start[0,j]

        self.opt_grid.atom_coords = self.opt_grid.atom_coords.detach().cpu()
        self.opt_grid.values = self.opt_grid.values.detach()

        if verbose:
            print('stopped on iter:', i+1)
            print('\nfinal loss:', self.losses[-1])

            coords_true = self.coords_true
            coords_pred = grid_pred.atom_coords[0, atom_pairs[:,0]]

            print('\nPRE-OPTIMIZE:')
            print('max distance from true atom: %.4f A'%np.linalg.norm(coords_pred - coords_true, axis=-1).max())
            print('mean distance from true atom: %.4f A'%np.linalg.norm(coords_pred - coords_true, axis=-1).mean())


            coords_pred = self.opt_grid.atom_coords[0,atom_pairs[:,0]]

            print('\nOPTIMIZED:')
            print('max distance from true atom: %.4f A'%np.linalg.norm(coords_pred - coords_true, axis=-1).max())
            print('mean distance from true atom: %.4f A'%np.linalg.norm(coords_pred - coords_true, axis=-1).mean())
            
        return self.opt_grid.atom_coords

    
    def fit_optimize(self, 
                     target_grid, 
                     coords_start=None, 
                     types_start=None,
                     boxes_mod=None,
                     max_iter=10, 
                     second_pass=False, 
                     amp_thresh=0.3, 
                     dist_thresh=0.9,
                     fit_kernel_size=3, 
                     opt_steric=False, 
                     verbose=False, 
                     **kwargs):
        '''Runs fit_density and opitimize_coordinates.'''

        ## FIRST PASS
        self.fit_density(target_grid=target_grid.copy(),
                          fit_kernel_size=fit_kernel_size,
                          amp_thresh=amp_thresh,
                          dist_thresh=dist_thresh, 
                          coords_start=coords_start, 
                          types_start=types_start,
                          boxes_mod=boxes_mod,
                          verbose=verbose
                          )


        self.conv_gauss = self.conv_gauss.cuda()
        dgrid_pred = self.get_pred_grid()

        coords_pred = self.optimize_coordinates(dgrid_pred, target_grid.values, max_iter=max_iter, opt_steric=opt_steric, **kwargs)[0].clone()

        if second_pass:
            #### SECOND PASS
            self.opt_grid.atom_coords = coords_pred
            self.opt_grid.encode_coords2grid()
            self.opt_grid.values = self.conv_gauss(self.opt_grid.values.to(self.conv_gauss.weight.device))


            self.fit_density(target_grid=self.opt_grid.copy(),
                              coords_start=[[self.target_grid.angstroms2voxels(x).unsqueeze(0)] for x in coords_pred], 
                              types_start=[x for x in self.types_pred], 
                              density_start=self.opt_grid.values, 
                              fit_kernel_size=fit_kernel_size,
                              amp_thresh=amp_thresh,
                              dist_thresh=dist_thresh,
                              boxes_mod=boxes_mod,
                              verbose=verbose
                              )

            dgrid_pred = self.get_pred_grid()
            coords_pred = self.optimize_coordinates(dgrid_pred, target_grid.values, 
                                                    max_iter=max_iter//2, 
                                                    opt_steric=opt_steric,
                                                    fix_coords=[i for i in range(len(coords_start))],
                                                    **kwargs)[0].clone()

        types_pred = self.opt_grid.atom_types[0]
        return coords_pred, types_pred
    

class MolFitter(object):


    def __init__(self, dfit, molmaker, debug=False) -> None:
        self.debug = debug
        self.dfit = dfit
        self.molmaker = molmaker
        self.atomtyper = AtomTyperDefault()
        '''Class for fitting molecules to density grids.'''

    def mol_from_coords_types(self, atom_types, atom_coords, bonddiv=1, add_hs=False, verbose=False):

        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.numpy()
        if isinstance(atom_coords, torch.Tensor):
            atom_coords = atom_coords.numpy()
        
        mol = ob.OBMol()
        mol.BeginModify()

        for i in range(len(atom_types)):
            obatom = mol.NewAtom()
            obatom.SetAtomicNum(int(atom_types[i]))
            pos = list(atom_coords[i].astype(float)/bonddiv)
            obatom.SetVector(*pos)


        mol.ConnectTheDots()

        if verbose:
            print('atoms', mol.NumAtoms())
            print('hydrogens', mol.NumAtoms()-mol.NumHvyAtoms())
            print('bonds', mol.NumBonds())

        mol.PerceiveBondOrders()
        mol.AddHydrogens()
        mol.FindAngles()
        mol.FindTorsions()
        
        mol.EndModify()

        pybelmol = pybel.Molecule(mol)
        pybelmol.write('sdf', '/tmp/tmp.sdf', overwrite=True)
        rdmol = next(Chem.SDMolSupplier('/tmp/tmp.sdf'))
        
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "smi")

        smiles = obConversion.WriteString(mol)

        if add_hs:
            rdmol = Chem.rdmolops.AddHs(rdmol)
            AllChem.EmbedMolecule(rdmol)

        return rdmol, smiles
    
    def new_mol_from_rdmol(self, mol, bonddiv=1, add_hs=False, verbose=False):
        mol = Mol2(mol)
        return self.mol_from_coords_types(mol.get_atom_nums(), mol.get_coords(), bonddiv, add_hs, verbose)
    


    def get_largest_fragment(self, mol : Chem.Mol, return_frags : bool = False, verbose : bool = False):
        """Separate and return largest fragment in molecule. Fragment is defined as largest connected substructure. If return_frags is true, returns remaining fragments as a separate Chem.Mol
        """    
        mol_clean = copy(mol)

        smi = Chem.MolToSmiles(mol_clean)
            
        frags = smi.strip().split('.')
        
        if verbose:
            print('fragments:', frags)
        
        while len(frags) > 1:
            f = frags.pop(np.argmin([len(x) for x in frags]))
            if verbose:
                print('Removing:', f)
            mol_clean = Chem.rdmolops.DeleteSubstructs(mol_clean, Chem.MolFromSmarts(f), onlyFrags=True)
            
        smi_clean = frags[0]
        
        if verbose:
            print('\nremaining:', smi_clean)
            
        if return_frags:
            mol_f = copy(mol)
            mol_f = Chem.rdmolops.DeleteSubstructs(mol_f, Chem.MolFromSmarts(smi_clean), onlyFrags=True)
            return mol_clean, mol_f
        else:
            return mol_clean


    def connect_mol_fragments(self, m, dist_thresh=3., check_valence=True, verbose=False):
        '''Connects fragments in a molecule if within distance threshold. 
        
        For each fragment: find closest atom on other fragments that are within distance threshold 
        for which valence is not full or has at least one higher order bond. If valence is full, but 
        atom has a higher order bonds, reduce bond order of one of those bonds. Then add a new bond between atoms.
        If no such atoms exist for a given fragment, leave that fragment unchanged.
        '''

        if m is None:
            return m
        
        if len(m.GetAtoms()) == 0:
            return m
        
        ptable = Chem.rdchem.GetPeriodicTable()

        m = Chem.RWMol(m)
        if verbose:
            print('\nmol sanitize', Chem.SanitizeMol(m, catchErrors=True) == 0)
        Chem.Kekulize(m, clearAromaticFlags=True)

        dmat = Chem.rdmolops.Get3DDistanceMatrix(m)
        frag_idxs = [list(x) for x in Chem.rdmolops.GetMolFrags(m)]
        n_skip = 0

        nwhile = 0
        while len(frag_idxs) > 1:
            nwhile += 1
            dmin = 100
            pairs = []
            dmins = []

            # get bondable atom pairs (within dist thresh)
            for i in frag_idxs[0]:
                for frag2 in frag_idxs[1:]:
                    for j in frag2:
                        if dmat[i,j] < dist_thresh:
                            dmins.append(dmat[i,j])
                            pairs.append([i,j])

            if len(dmins) == 0: # no bondable atom pairs for this fragment
                frag_idxs.pop(0)
                n_skip += 1
                continue

            dmins = np.array(dmins)
            pairs = np.array(pairs, dtype=int)

            # sort by distance
            isort = np.argsort(dmins)
            dmins = dmins[isort]
            pairs = pairs[isort]


            # attempt to bond closest valid pair of atoms
            for i in range(len(dmins)):
                ai = int(pairs[i,0])
                aj = int(pairs[i,1])

                bond_success = False
                for idx in [ai,aj]:
                    atom = m.GetAtomWithIdx(idx)
                    def_valence = list(ptable.GetValenceList(atom.GetAtomicNum()))
                    bond_orders = [0]+[b.GetBondTypeAsDouble() for b in atom.GetBonds()]
                    
                    if check_valence:
                        if sum(bond_orders) < max(def_valence):
                            bond_success = True
                            continue
                        if sum(bond_orders) >= max(def_valence):
                                bond_success = False
                                break
                        if sum(bond_orders) == max(def_valence) and  max(bond_orders) > 1.:
                            for b in atom.GetBonds():
                                if b.GetBondTypeAsDouble() > 1.:
                                    bond_success = True
                                    b.SetBondType(Chem.BondType.SINGLE)
                                    break  # only need to lower one bond order
                            if not bond_success:
                                break
                    else:
                        bond_success = True
                        break

                
                if not bond_success:
                    continue
                    
                if dmins[i] == 0:
                    if verbose:
                        print('Atoms are located on top of eachother!')

                if verbose:
                    print('adding bond:', pairs[i], 'dist:', dmins[i])

                m.AddBond(ai, aj, Chem.BondType.SINGLE)
                frag_idxs = [list(x) for x in Chem.rdmolops.GetMolFrags(m)]
                dmat = Chem.rdmolops.Get3DDistanceMatrix(m)
                break

            if not bond_success:
                frag_idxs.pop(0)
                n_skip += 1
            continue

        return Chem.Mol(m)
    


    def fit_logits(self, logits, model, coords_start=None, types_start=None, boxes_mod=None, grids=None, version='v2', upsample_res2=True, set_props=True, verbose=False):
        '''Fits model output logits to a molecule.'''
        output = model.hvae_model.dist_from_output(logits[0])
        output2 = model.hvae_model.dist_from_output(logits[1])
        
        res1_grid = output.sample_vals_mean().detach().cpu().clone()
        res2_grid = output2.sample_vals_mean().detach().cpu().clone()

        self.conv_gauss0 = self.dfit.conv_gauss.re_init(n_channels=1)
        
        if grids is None:
            grids = model.init_lig_grids()
        lig_grid, lig_grid2 = grids
            
        lig_grid.values = copy(res1_grid)
            
        if upsample_res2:
            res2_grid = torch.nn.functional.interpolate(res2_grid.float(), scale_factor=2, mode='trilinear')
            lig_grid2.values = copy(res2_grid)
            lig_grid2.resolution = lig_grid.resolution
        else:
            lig_grid2.values = res2_grid
            lig_grid2.resolution = lig_grid.resolution*2 


        if coords_start is not None:
            if isinstance(coords_start, np.ndarray):
                coords_start = torch.from_numpy(coords_start)
            if isinstance(coords_start, torch.Tensor):
                coords_start = [[lig_grid2.angstroms2voxels(x).unsqueeze(0)] for x in coords_start]


        if types_start is not None:
            if isinstance(types_start, (np.ndarray, torch.Tensor)):
                if types_start.shape[-1] == 1024:
                    raise NotImplementedError  # 1hot encoding not implemented
                else:
                    types_start = types_start.tolist()

            if isinstance(types_start, list):
                if isinstance(types_start[0], (int,float)):
                    types_start = [[x] for x in types_start]
                for i in range(len(types_start)):
                    types_start[i] = [lig_grid2.get_type_channel(int(t)) for t in types_start[i]]

        if version == 'v1':
            mol_pred, coords_pred, types_pred = self.fit_grids_v1((lig_grid, lig_grid2), coords_start=coords_start, types_start=types_start, boxes_mod=boxes_mod, verbose=verbose)
        elif version == 'v2':
            mol_pred, coords_pred, types_pred = self.fit_grids_v2((lig_grid, lig_grid2), coords_start=coords_start, types_start=types_start, boxes_mod=boxes_mod, verbose=verbose)
        else:
            raise Exception(f'Invalid fit_grids func version: {version}""')
        
        if set_props:
            mol_pred = self.molmaker.new_mol_from_rdmol(mol_pred)
            mol_pred = self.molmaker.set_mol_props(mol_pred)
        else:
            clean_nonring_aromaticity(mol_pred)
        return mol_pred, coords_pred, types_pred, (res1_grid, res2_grid)


    def fit_grids_v1(self, lig_grids, amp_thresh=0.3, dist_thresh=0.9, prune_dist_thresh=0.8, coords_start=None, types_start=None, boxes_mod=None, verbose=False):
        '''Fits a molecule to a density grid (v1).'''
        lig_grid, lig_grid2 = lig_grids
        dgrid_l = lig_grid.copy()
        dgrid_l.values = dgrid_l.values.float()

        idxs_fix = []
        if coords_start is not None:
            coords_start0 = copy(coords_start)
            idxs_fix = list(range(len(coords_start0)))

        
        types_start_res2 = types_start
        types_start_res1 = [[0] for _ in types_start] if types_start is not None else None

        
        if verbose:
            print('max grid val:', dgrid_l.values.max())
        try:
            coords_pred, types_pred = self.dfit.fit_optimize(target_grid=dgrid_l, 
                                                        max_iter=10, 
                                                        second_pass=False, 
                                                        amp_thresh=amp_thresh, 
                                                        dist_thresh=dist_thresh,
                                                        opt_steric=False, 
                                                        coords_start=coords_start,
                                                        types_start=types_start_res1,
                                                        boxes_mod=boxes_mod,
                                                        )
        except Exception as e:
            print('fit_optimize error')
            raise e
            return None, coords_pred, None

        opt_grid = self.dfit.opt_grid

        ## deal with atoms that have multiple types. (greedy approach)
        types_multi = np.arange(len(types_pred))[(types_pred.sum(axis=-1) > 1).flatten()]
        if len(types_multi) > 0:
            for i in types_multi:
                types_pred[i][types_pred[i].nonzero()[1:]] = False


        coords_pred, types_pred = opt_grid.flatten_coords_types(coords_pred, types_pred)

        coords_pred, idxs_keep = prune_coords(coords_pred, dist_thresh=prune_dist_thresh, coords_fix=idxs_fix)

        # get types from res2 grid
        types_pred = self.fit_types_res2(lig_grid2, coords_pred)
        
        if types_pred is None:
            mol_pred = None
        else:
            self.molmaker.debug = False
            mol_pred = self.molmaker.make_new_mol(coords_pred, types_pred, no_fragment=True)
            
        self.molmaker.debug = False
        mol_pred = self.molmaker.make_new_mol(coords_pred, types_pred, no_fragment=True)
        return mol_pred, coords_pred, types_pred


    def fit_grids_v2(self, lig_grids, amp_thresh=0.3, dist_thresh=0.9, prune_dist_thresh=0.8, coords_start=None, types_start=None, boxes_mod=None, verbose=False):
        '''Fits a molecule to a density grid (v2).'''
        debug = self.debug
        lig_grid, lig_grid2 = lig_grids
        self.dfit.target_grid = lig_grid.copy()
        self.dfit.target_grid.values = self.dfit.target_grid.values.float()
        if coords_start is not None:
            coords_start0 = copy(coords_start)
            idxs_fix = list(range(len(coords_start0)))
            coords_start0_ang = lig_grid2.voxels2angstroms(torch.stack([x[0] for x in coords_start0])).reshape(-1,3)
        else:
            idxs_fix = []
        
        gauss_var = self.dfit.conv_gauss.var
        if verbose:
            print('max grid val:', self.dfit.target_grid.values.max())

        fit_kernel_size=1
        max_iter=20

        

        if types_start is None:
            types_start = []
        types_start_res2 = types_start
        types_start_res1 = [[0] for _ in types_start]

        num_opt_iter = 4
        for opt_iter in range(num_opt_iter):
        ######## fit optimize
            if opt_iter == 0:
                self.dfit.conv_gauss = self.conv_gauss0.re_init(var=gauss_var*1.)
                self.dfit.conv_gauss.weight.data[self.dfit.conv_gauss.weight.data > 0.9] *= 1.9
                opt_bond_dist = False
                opt_steric=False
    #             dfit.conv_gauss = conv_gauss0.re_init(var=gauss_var*1.)
    #             dfit.conv_gauss.weight.data[dfit.conv_gauss.weight.data < 0.9] /= 1.2

                coords_pred, types_pred = self.dfit.fit_density(target_grid=self.dfit.target_grid.copy(),
                                                                fit_kernel_size=fit_kernel_size,
                                                                amp_thresh=amp_thresh,
                                                                dist_thresh=dist_thresh, 
                                                                coords_start=coords_start, 
                                                                types_start=types_start_res1,
                                                                coords_fix=idxs_fix,
                                                                boxes_mod=boxes_mod,
                                                                verbose=verbose
                                                                )
                                                
                
                if debug:
                    print('\n\nopt %d-0:'%opt_iter, (coords_pred[:len(coords_start0)] == coords_start0_ang).sum())

    #             dfit.conv_gauss = conv_gauss0.re_init(var=gauss_var*1.)
    #             dfit.conv_gauss.weight.data[dfit.conv_gauss.weight.data > 0.9] *= 1.9

            elif opt_iter < num_opt_iter-2:
                opt_bond_dist = False
                opt_steric=False
                self.dfit.conv_gauss = self.conv_gauss0.re_init(var=gauss_var*1.)
                self.dfit.conv_gauss.weight.data[self.dfit.conv_gauss.weight.data < 0.9] /= 1.25
    #             dfit.conv_gauss.weight.data /= 1.25

                self.dfit.opt_grid.atom_coords = coords_pred
                self.dfit.opt_grid.encode_coords2grid()
                self.dfit.opt_grid.values = self.dfit.conv_gauss(self.dfit.opt_grid.values.to(self.dfit.conv_gauss.weight.device))
                self.dfit.types_pred = [[self.dfit.target_grid.channel_nums.index(x)] for x in types_pred.tolist()]

                coords_pred, types_pred = self.dfit.fit_density(target_grid=self.dfit.target_grid.copy(),
                                          coords_start=[[self.dfit.target_grid.angstroms2voxels(x).unsqueeze(0)] for x in coords_pred], 
                                          types_start=[x for x in self.dfit.types_pred], 
                                          density_start=self.dfit.opt_grid.values, 
                                          fit_kernel_size=fit_kernel_size,
                                          amp_thresh=amp_thresh,
                                          dist_thresh=dist_thresh,
                                          coords_fix=idxs_fix,
                                          boxes_mod=boxes_mod,
                                          )
                
                if debug:
                    print('opt %d-1:'%opt_iter, (coords_pred[:len(coords_start0)] == coords_start0_ang).sum())

                self.dfit.conv_gauss = self.conv_gauss0.re_init(var=gauss_var*1.)
                self.dfit.conv_gauss.weight.data[self.dfit.conv_gauss.weight.data > 0.8] *= 1.4
            else:
                opt_bond_dist = False
                opt_steric=False
                prune_dist_thresh=0.9
                if opt_iter == num_opt_iter-1:
                    opt_steric=False
                # optimize with standard conv_gauss
                self.dfit.conv_gauss = self.conv_gauss0.re_init(var=gauss_var*1.)

                self.dfit.opt_grid.atom_coords = coords_pred
                self.dfit.opt_grid.encode_coords2grid()
                self.dfit.opt_grid.values = self.dfit.conv_gauss(self.dfit.opt_grid.values.to(self.dfit.conv_gauss.weight.device))
                self.dfit.types_pred = [[self.dfit.target_grid.channel_nums.index(x)] for x in types_pred.tolist()]



            dgrid_pred = self.dfit.get_pred_grid()
            coords_pred = self.dfit.optimize_coordinates(dgrid_pred, 
                                                    self.dfit.target_grid.values, 
                                                    max_iter=20, 
                                                    opt_steric=opt_steric, 
                                                    opt_steric_w=2e1,
                                                    steric_dist=1.3,
                                                    opt_bond_dist=opt_bond_dist,
                                                    opt_bond_w=1e-2,
                                                    fix_coords=idxs_fix,
                                                )[0].clone()
            
            if debug:
                    print('opt %d:'%opt_iter, (coords_pred[:len(coords_start0)] == coords_start0_ang).sum())

            types_pred = self.dfit.opt_grid.atom_types[0]

            try:
                ## deal with atoms that have multiple types. (greedy approach)
                types_multi = (types_pred.sum(axis=-1) > 1).flatten()
                if types_multi.sum() > 0:
                    for i in np.nonzero(types_multi)[0]:
                        types_pred[i][types_pred[i].nonzero()[1:]] = False
            except Exception as e:
                print('types_multi', types_multi)
                print('len(types_pred):', len(types_pred))
                print('(types_pred.sum(axis=-1) > 1).flatten():', (types_pred.sum(axis=-1) > 1).flatten())
                raise e

            if debug:
                    print('opt %d.1:'%opt_iter, (coords_pred[:len(coords_start0)] == coords_start0_ang).sum())

            coords_pred, types_pred = self.dfit.opt_grid.flatten_coords_types(coords_pred, types_pred)
            if debug:
                    print('opt %d.2:'%opt_iter, (coords_pred[:len(coords_start0)] == coords_start0_ang).sum())

            if opt_iter < num_opt_iter-1:
                #### PRUNE
                # sort coords by voxel amplitude
                pred_grid = self.dfit.get_pred_grid()
                pred_grid.atom_coords = coords_pred
                pred_grid.atom_types = types_pred
                grid_vals = self.dfit.target_grid.get_atom_grid_vals(grid_coords=pred_grid.atom_coords_grid,
                                                                grid_types=pred_grid.atom_channels_flat,
                                                                use_types=True)[0]
                
                grid_vals_sort, idxs_sort = grid_vals.sort(descending=True)
                idxs_sort_keep = idxs_sort

                # remove atoms below amplitude thresh
                idxs_remove = (grid_vals_sort < 0.2)
                # print('idxs_remove', idxs_remove.shape)
                for idx in idxs_fix:
                    idxs_remove[idxs_sort.tolist().index(idx)] = False
                idxs_sort_keep = idxs_sort[~idxs_remove]
                if debug:
                    print('opt %d.30:'%opt_iter, (coords_pred[:len(coords_start0)] == coords_start0_ang).sum())

                coords_pred = coords_pred[idxs_sort_keep]
                types_pred = types_pred[idxs_sort_keep]

                if debug:
                    print('opt %d.31:'%opt_iter, (coords_pred[np.argsort(idxs_sort_keep)][:len(coords_start0)] == coords_start0_ang).sum())

                # prune coords within distance threshold of other atoms, starting with lowest amplitude value first
                coords_pred, idxs_keep = prune_coords(coords_pred, dist_thresh=prune_dist_thresh, coords_fix=idxs_fix)
                types_pred = types_pred[idxs_keep]

                idxs_unsort = np.argsort(idxs_sort_keep[idxs_keep])
                coords_pred = coords_pred[idxs_unsort]
                types_pred = types_pred[idxs_unsort]

                self.dfit.opt_grid.atom_coords = coords_pred
                self.dfit.opt_grid.atom_types = types_pred

                self.dfit.types_pred_1hot = self.dfit.opt_grid.atom_types
                self.dfit.coords_pred = self.dfit.opt_grid.atom_coords
                
                if debug:
                    print('opt %d.32:'%opt_iter, (coords_pred[:len(coords_start0)] == coords_start0_ang).sum())
            ########


        # get types from res2 grid
        if coords_start is not None:
            coords_start_ang = lig_grid2.voxels2angstroms(torch.stack([x[0] for x in coords_start0])).reshape(-1,3).numpy()
            types_pred = self.fit_types_res2(lig_grid2, coords_pred, types_start=types_start_res2, coords_start=coords_start_ang)
        else:
            types_pred = self.fit_types_res2(lig_grid2, coords_pred, types_start=None)
        
        if types_pred is None:
            mol_pred = None
        else:
            self.molmaker.debug = False
            mol_pred = self.molmaker.make_new_mol(coords_pred, types_pred, no_fragment=True)
        return mol_pred, coords_pred, types_pred
    


    def fit_types_res2(self, lig_grid2, coords_pred, types_start=None, coords_start=None):
        '''Fits atom types to atomic coordinates given an atom types density grid.'''
        self.amp_thresh = 0.15
        lig_grid2.atom_coords = coords_pred
        coords_grid = lig_grid2.atom_coords_grid[0]
        coord_weights, cround = lig_grid2.get_encode_weights(grid_coords=coords_grid,
                                                            grid=lig_grid2.values[0],
                                                            encoding='softcube')
        if len(cround) < len(coords_pred):
            # some coords are out of grid bounds
            print('Ref mol has atoms out of bounds!')
            return None

        if types_start is not None:
            assert coords_start is not None, 'Must provide `coords_start` if `types_start` is not None.'
            assert len(coords_start) == len(types_start), 'lengths of `coords_start` and `types_start` don\'t match'
            coords = lig_grid2.voxels2angstroms(coords_grid).numpy().reshape(-1,3)
            dmat = cdist(coords, coords_start)
            idxs_start = [(i, x.argmin()) for i, x in enumerate(dmat)]
            idxs_start = [x for x in idxs_start if dmat[x[0], x[1]] < 0.5]
            idx_replace = [x[0] for x in idxs_start]


        types_pred = []
        for j, cr in enumerate(cround):
            if (types_start is not None) and (j in idx_replace):
                k = idx_replace.index(j)
                ch = types_start[idxs_start[k][1]]
            else:
                w = coord_weights[j]
                cr = cr.int()
                tvec = lig_grid2.values[0,:, cr[0]-1:cr[0]+2, cr[1]-1:cr[1]+2, cr[2]-1:cr[2]+2]
                tvec = (tvec*w).sum(dim=(-1,-2,-3))
                ch = (tvec >= self.amp_thresh).nonzero().flatten()
            if len(ch) == 0:
                types_pred.append(torch.zeros(self.atomtyper.ntypes, dtype=bool)) # empty types vector
            else:
                tnums = self.atomtyper.nums2vec(list(itertools.chain.from_iterable([lig_grid2.channel_nums[i] for i in ch])))
                
                # check for multiple element types. choose highest amplitude
                if tnums[:128].sum() > 1:
                    tnums[:128] = 0
                    amax = np.argmax(tvec[lig_grid2.channels_elements_only])
                    tnums[lig_grid2.channel_nums[amax][0]] = 1
                types_pred.append(tnums)

        return torch.stack(types_pred)


class MolMaker(object):
    def __init__(self, bond_dist_max=3.5, bond_stretch_max=1.8, bond_angle_min=50, debug=False):
        '''Class for fitting bonds to a set of atoms with coordinates and types and setting molecular properties.'''
        self.bond_dist_max = bond_dist_max
        self.bond_stretch_max = bond_stretch_max
        self.bond_angle_min = bond_angle_min
        self.debug = debug
        
        self.ptable = Chem.rdchem.GetPeriodicTable()
        self.atomtyper = AtomTyperDefault()

    def add_bonds_within_dist(self, mol):
        bond_dist_max = self.bond_dist_max
        dmat = Chem.Get3DDistanceMatrix(mol)

        for i in range(mol.GetNumAtoms()):
            for j in range(i+1, mol.GetNumAtoms()):
                if (dmat[i,j] < bond_dist_max) and (mol.GetBondBetweenAtoms(i,j) is None):
                    mol.AddBond(i,j, Chem.BondType.SINGLE) 
        return mol


    def get_atom_bond_stretch(self, atom, absolute=False):
        mol = atom.GetOwningMol()
        bond_stretch = []
        for bond in atom.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            stretch = self.get_bond_stretch(bond, absolute)
            bond_stretch.append([stretch, i, j])
        bond_stretch = sorted(bond_stretch, key=lambda x: x[0])
        if not absolute:
            bond_stretch = list(reversed(bond_stretch))
        return bond_stretch

    
    def get_bond_stretch(self, bond, absolute=False):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ai, aj = bond.GetBeginAtom(), bond.GetEndAtom()
        conf = ai.GetOwningMol().GetConformer(0)
        length_default = self.ptable.GetRcovalent(ai.GetAtomicNum()) + self.ptable.GetRcovalent(aj.GetAtomicNum())
        bond_length = Chem.rdMolTransforms.GetBondLength(conf, i, j)
        stretch = bond_length/length_default
        if absolute:
            stretch = abs(np.log2(stretch))
        return stretch


    def get_atom_bond_angles(self, atom):
        mol = atom.GetOwningMol()

        ai = atom.GetIdx()
        angles = []
        nbrs = atom.GetNeighbors()
        if len(nbrs) <= 1:
            return angles

        conf = mol.GetConformer(0)
        for j in range(len(nbrs)):
            for k in range(j+1, len(nbrs)):
                aj = nbrs[j].GetIdx()
                ak = nbrs[k].GetIdx()
                angle = Chem.rdMolTransforms.GetAngleDeg(conf, aj,ai,ak)
                angles.append((angle, (aj,ai,ak)))
        return list(sorted(angles, key=lambda x: x[0]))

    def get_atom_valence(self, atom):
        return int(sum([0]+[b.GetBondTypeAsDouble() for b in atom.GetBonds()]))


    def reset_bond_orders_aromatic(self, mol):
        mol = Chem.RWMol(mol)
        for atom in mol.GetAtoms():
            max_valence = self.get_atom_max_valence(atom)
            val = self.get_atom_valence(atom)
            num_bonds_remove = val - max_valence

            if num_bonds_remove > 0:
                if any([b.GetIsAromatic() for b in atom.GetBonds()]):
                    for b in atom.GetBonds():
                        b.SetBondType(Chem.BondType.SINGLE)
                        b.SetIsAromatic(False)
                    atom.SetIsAromatic(False)
        mol = clean_ring_aromaticity(mol)
        return Chem.Mol(mol)

        

    def remove_bonds_cleanup_valence(self, mol, stretched_only=False, no_fragment=True):
        mol = Chem.RWMol(mol)

        for atom in mol.GetAtoms():
            max_valence = self.get_atom_max_valence(atom)
            val = self.get_atom_valence(atom)

            num_bonds_remove = val - max_valence

            if num_bonds_remove > 0:
                bonds_stretch = self.get_atom_bond_stretch(atom)
                i_removed = 0
                k = 0
                while i_removed < num_bonds_remove:
                    stretch, i0, i1 = bonds_stretch[k]
                    if not stretched_only or stretch > 1.5:
                        if no_fragment and self.check_bond_will_fragment(mol.GetBondBetweenAtoms(i0, i1)):
                            if self.debug:
                                print('Skipping to avoid fragmentation bond (%d, %d) with stretch %.2f'%(i0, i1, stretch))
                        else:
                            if self.debug:
                                print('Removing bond (%d, %d) with stretch %.2f'%(i0, i1, stretch))
                            mol.RemoveBond(i0, i1)
                            i_removed += 1
                    k += 1
                    if k == len(bonds_stretch):
                        break
        return Chem.Mol(mol)


    def get_atom_max_valence(self, atom):
        max_valence = self.ptable.GetDefaultValence(atom.GetAtomicNum())
        if atom.GetAtomicNum() == 16:
            if sum([nbr.GetAtomicNum()==8 for nbr in atom.GetNeighbors()]) >= 2:
                if self.debug:
                    print('Resetting sulfur valence to 6')
                max_valence = 6
        return max_valence



    def remove_bonds_stretch(self, mol, no_fragment=True):
        bond_stretch_max = self.bond_stretch_max

        mol = Chem.RWMol(mol)
        bonds_remove = []
        for bond in list(mol.GetBonds()):        
            stretch = self.get_bond_stretch(bond)
            if stretch > bond_stretch_max:
                i0, i1 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if no_fragment and self.check_bond_will_fragment(bond):
                    if self.debug:
                        print('Skipping to avoid fragmentation bond (%d, %d) with stretch %.2f'%(i0, i1, stretch))
                else:
                    if self.debug:
                        print('Removing bond (%d, %d) with stretch %.2f'%(i0, i1, stretch))
                    bonds_remove.append((i0, i1))
                    mol.RemoveBond(i0, i1)
        return Chem.Mol(mol)



    def remove_bonds_bad_angle(self, mol, no_fragment=True):
        bond_angle_min = self.bond_angle_min
        mol = Chem.RWMol(mol)

        for atom in mol.GetAtoms():
            bond_angles = self.get_atom_bond_angles(atom)
            bond_angles_bad = [x for x in bond_angles if x[0] < bond_angle_min]
            if len(bond_angles_bad) == 0:
                continue
            
            angles_skip = []
            while len(bond_angles_bad) > 0:
                conf = mol.GetConformer(0)
                angle, (i,j,k) = bond_angles_bad[0]
                bonds = [(mol.GetBondBetweenAtoms(i,j), i, j), (mol.GetBondBetweenAtoms(j,k), j, k)]
                bond_stretches = [(self.get_bond_stretch(x[0], absolute=False) , x[1], x[2]) for x in bonds if x[0] is not None]

                if len(bond_stretches) > 0:
                    bond_stretches = reversed(sorted(bond_stretches, key=lambda x: x[0]))
                    
                    remove_success = False
                    for (stretch, i0, i1) in bond_stretches:
                        if no_fragment and self.check_bond_will_fragment(mol.GetBondBetweenAtoms(i0, i1)):
                            if self.debug:
                                print('Skipping to avoid fragmentation bond (%d, %d) with stretch %.2f'%(i0, i1, stretch))
                        else:
                            if self.debug:
                                print('Removing bond (%d, %d) with angle %.2f'%(i0, i1, angle))
                            mol.RemoveBond(i0, i1)   
                            remove_success = True
                            break
                    if not remove_success:                        
                        angles_skip.append((i,j,k))
                else:
                    if self.debug:
                        print('WARNING: all bonds are None: indices (%d, %d, %d)'%(i,j,k))
                
                bond_angles = self.get_atom_bond_angles(atom)
                bond_angles_bad = [x for x in bond_angles if x[0] < bond_angle_min]
                bond_angles_bad = [x for x in bond_angles_bad if not any([x[1] == y for y in angles_skip])]
        return Chem.Mol(mol)


    def check_bond_will_fragment(self, bond):
        mol = bond.GetOwningMol()
        if len(bond.GetBeginAtom().GetBonds())==1 and len(bond.GetEndAtom().GetBonds())==1:
            return True
        else:
            i0 = bond.GetBeginAtomIdx()
            i1 = bond.GetEndAtomIdx()
            n_frags_start = len(Chem.rdmolops.GetMolFrags(mol))
            mol2 = copy(mol)
            mol2.RemoveBond(i0, i1)
            return n_frags_start != len(Chem.rdmolops.GetMolFrags(mol2))


    def rdmol_from_coords_types(self, coords, types):
        mol = Chem.RWMol()
        idxs_valid = (types[:, :128].sum(axis=-1) > 0)
        coords = coords[idxs_valid]
        types = types[idxs_valid]
        conf = Chem.Conformer(len(coords))
        for i, c in enumerate(coords):
            ts = types[i].nonzero()
            if len(ts) == 0:
                continue
            anums = [x[0] for x in ts if self.atomtyper.is_element(x[0])]
            if len(anums) == 0:
                continue
            
            atom = Chem.Atom(int(anums[0]))
            for t in ts:
                t = t[0]
                if self.atomtyper.is_atom_feature(t):
                    if self.atomtyper.n2t[int(t)] == 'FChargePos':
                        atom.SetFormalCharge(1)
                    if self.atomtyper.n2t[int(t)] == 'FChargeNeg':
                        atom.SetFormalCharge(-1)
                    if self.atomtyper.n2t[int(t)] == 'Aromatic':
                        atom.SetIsAromatic(True)
                    if self.atomtyper.n2t[int(t)] == 'HBA':
                        pass
                    if self.atomtyper.n2t[int(t)] == 'HBD':
                        pass
            
            mol.AddAtom(atom)
            conf.SetAtomPosition(i, tuple(c))
        mol.AddConformer(conf)
        mol = self.add_bonds_within_dist(mol)
        return Chem.Mol(mol)


    def retry_add_bonds(self, mol):
        m = Chem.RWMol(mol)
        m = self.add_bonds_within_dist(m)
        m = self.remove_bonds_cleanup_valence(m)
        m = self.remove_bonds_bad_angle(m)
        m = self.remove_bonds_stretch(m)
        return Chem.Mol(m)
        

    def make_new_mol(self, atom_coords, atom_types, no_fragment=True, debug=False):
        if isinstance(atom_coords, torch.Tensor):
            atom_coords = atom_coords.numpy()
        
        if debug:
            mol_steps = []
        atom_coords = atom_coords.reshape(-1,3)

        mol = self.rdmol_from_coords_types(atom_coords, atom_types)
        if debug: mol_steps.append(mol)

        mol = self.remove_bonds_cleanup_valence(mol, no_fragment=no_fragment)
        if debug: mol_steps.append(mol)

        mol = self.remove_bonds_bad_angle(mol, no_fragment=no_fragment)
        if debug: mol_steps.append(mol)

        mol = self.remove_bonds_stretch(mol, no_fragment=no_fragment)
        if debug: 
            mol_steps.append(mol)
            return mol_steps
        else:
            return mol
        
    def new_mol_from_rdmol(self, mol, no_fragment=True):
        mol = Mol2(mol)
        ts = mol.get_atom_nums().reshape(-1,1)
        ts = self.atomtyper.nums2vec(ts)
        return self.make_new_mol(mol.get_coords(), ts, no_fragment=no_fragment)
    

    def set_mol_props_ob(self, mols):
        return_single = False
        if not isinstance(mols, list):
            mols = [mols]
            return_single = True

        mols2 = [copy(m) for m in mols]

        mols_clean0 = [copy(m) for m in mols2]

        ob0 = [rdmol2obmol(m) for m in mols2]
        [perceive_bond_orders_ob(m) for m in ob0]
        [perceive_hybridization_ob(m) for m in ob0]
        [set_aromaticity_ob(m) for m in ob0]
        mols2 = [obmol2rdmol(m) for m in ob0]
        mols2 = [m if (len(Chem.DetectChemistryProblems(m)) == 0) else mols_clean0[i] for i,m in enumerate(mols2)]
        if return_single:
            mols2 = mols2[0]
        return mols2
        

    def set_mol_props_rdkit(self, mols):
        return_single = False
        if not isinstance(mols, list):
            mols = [mols]
            return_single = True

        mols2 = [copy(m) for m in mols]
        mols_clean0 = [copy(m) for m in mols2]

        [Chem.GetSSSR(m) for m in mols2]
        [clean_nonring_aromaticity(m) for m in mols2]
        mols2 = [set_aromaticity(m) for m in mols2]
        mols2 = [self.reset_bond_orders_aromatic(mol) for mol in mols2 if mol is not None]
        mols2 = [self.remove_bonds_cleanup_valence(mol) for mol in mols2 if mol is not None]
        [clean_nonring_aromaticity(m) for m in mols2]
        mols2 = [m if (len(Chem.DetectChemistryProblems(m)) == 0) else mols_clean0[i] for i,m in enumerate(mols2)]
        if return_single:
            mols2 = mols2[0]
        return mols2


    def set_mol_props(self, mols):
        return_single = False
        if not isinstance(mols, list):
            mols = [mols]
            return_single = True

        mols2 = self.set_mol_props_ob(mols)
        mols2 = self.set_mol_props_rdkit(mols2)
        [m.UpdatePropertyCache(strict=False) for m in mols2]
        if return_single:
            mols2 = mols2[0]
        return mols2
