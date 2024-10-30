import os, sys, glob
import numpy as np
import pandas as pd
from copy import copy
from os.path import join, dirname, basename
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
import subprocess

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast

from rdkit import Chem
from rdkit.Chem import AllChem

from .lightning import HVAEComplexSplit
from .molecules import BulkSDMolParser, MolFilter, MolParser, get_mol_stats, shift_mol, get_largest_fragment, write_mols_sdf, get_mol_center, transform_coordinates
from .molfitting import MolFitter, MolMaker, DensityFitter, rotate_coordinates, translate_coordinates
from .trainutils import Hparams
from .gridutils import rot3d, trans3d


def get_checkpoints_from_dir(run_dir):
    check_files = glob.glob(os.path.join(run_dir,'checkpoints/*'))
    check_files = sorted(check_files, key=lambda x: int(x.strip('.ckpt').split('step=')[1].replace('-v1','')))
    return check_files


def check_port_open(addr, port):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket.setdefaulttimeout(2.0)
    try:
        sock.connect((addr, int(port)))
        result = True
    except:
        result = False
    finally:
        sock.close()
    return (result == 0)


def init_process_group(port=6004):
    if not dist.is_initialized():
        addr = '127.0.0.1'
        while not check_port_open(addr, port):
            port += 1
            print('port', port)
        os.environ['MASTER_ADDR'] = addr
        print('Setting port to:', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=1)
    else:
        print('process group already initialized.')


def load_hvae_model(checkpath):
    print(checkpath)
    
    checkpoint = torch.load(checkpath, map_location='cpu')

    args = checkpoint['hyper_parameters']['args']
    model_class = eval(args.dict.get('model_class', 'HVAE'))
    print('\nmodel_class:', model_class)

    model = model_class.load_from_checkpoint(checkpath, strict=False)

    model.cuda()
    model.hvae_model.cuda()
    model.train()

    if 'zinc20_large_uniform.h5' in model.dataset.dataset_zinc.data_path:
        model.dataset.dataset_zinc.data_path = 'data/ZINC12_clean/zinc12_clean_100k.h5'
    return model


def format_zbetas_str(zbetas, nres=4, ngroup=4):
    """Generates a string that summarizes the list of zbetas given as input

    Args:
        zbetas: single value or list of values.
        nres: number of latent resolutions in the model
        ngroup: number of groups per latent resolution
    """
    if isinstance(zbetas, (float,int)):
        zstr = f'_{int(zbetas*10):d}'
    elif isinstance(zbetas, (list, np.ndarray)):
        zbetas = np.array(zbetas)
        if len(np.unique(zbetas)) == 1:
            zstr = f'_{int(zbetas[0]*10):d}'
        elif len(zbetas) == nres:
            zb_vals = zbetas
            zstr = ''
            for zb in zbetas:
                zstr += f'_{int(zb*10):d}'
        elif len(zbetas) == int(nres*ngroup):
            zb_vals = []
            for ri in range(nres):
                idxs = np.array([0,1,2,3]) + ri*ngroup
                if len(np.unique(zbetas[idxs])) == 1:
                    zb_vals.append([zbetas[idxs][0]])
                else:
                    zb_vals.append(zbetas[idxs].tolist())
                zstr = ''
                for v in zb_vals:
                    zstr += '_'+'-'.join([f'{int(x*10):d}' for x in v])
    return zstr

class MolGenerator(object):
    def __init__(self, checkpoint, model_id=None, random_rot=False, random_trans=False, ffopt=False) -> None:
        self.model_init = False
        self.random_rot = random_rot
        self.random_trans = random_trans
        self.model_id = model_id if model_id else basename(checkpoint).replace('.ckpt','')
        self.checkpoint = checkpoint
        self.ffopt = ffopt


    def initialize(self):
        if not self.model_init:
            init_process_group()
            print('Loading model...' , end='', flush=True)
            model = load_hvae_model(self.checkpoint)
            model = model.train()
            self.model = model
            print('done')

            self.conv_gauss = model.gauss_conv_layer_lig.cuda().re_init(n_channels=1)

            self.dfit = DensityFitter(self.conv_gauss, device='cpu', atom_size=7, conv_kernel_size=3)
            self.molmaker = MolMaker(bond_dist_max=3.5, bond_stretch_max=1.8, bond_angle_min=50, debug=False)
            self.molfitter = MolFitter(self.dfit, self.molmaker)
            self.model_init = True
        else:
            print('Already initialized. Skipping...')

    
    def get_save_path(self, savedir, zbetas, temps, n_samples):
        gen_name = 'prior'
        if isinstance(zbetas, (int,float)):
            zb = zbetas
        else:
            zb = sum(zbetas)
        if zbetas is not None and (zb > 0):
            gen_name = 'posterior'
            gen_name += format_zbetas_str(zbetas)
        print('gen_name:', gen_name)

        saveroot = savedir
        saveroot = join(savedir, gen_name)
        os.makedirs(saveroot, exist_ok=True)
        
        print('\nsaving generated mols to:', saveroot)
        with open(join(saveroot,'gen.txt'), 'w+') as f:
            f.write('n_samples: %s'%str(n_samples))
            f.write('\nzbetas: %s'%str(zbetas))
            f.write('\ntemps: %s'%str(temps))
            f.write('\nrandom_rot: %s'%str(self.random_rot))
            f.write('\nrandom_trans: %s'%str(self.random_trans))
        return saveroot


    def post_process_mol(self, mol, coords_pred, molfilter, transform=None):                      
        if mol is not None and len(Chem.DetectChemistryProblems(mol)) > 0:
            mol = None
        
        if mol is not None:
            try:
                Chem.GetSSSR(mol)
                mol = self.molfitter.connect_mol_fragments(mol, dist_thresh=3., verbose=False)
                mol = self.molfitter.get_largest_fragment(mol)
                if molfilter and not molfilter.check_mol(mol):
                    mol = None
            except:
                mol = None

        if transform is not None and mol is not None:
            if 'rot' in transform:
                coords_pred = rot3d(coords_pred, transform['rot'], inverse=True)
                rotate_coordinates(mol, transform['rot'], inverse=True)
                
            if 'trans' in transform:
                coords_pred = trans3d(coords_pred, transform['trans'], inverse=True)
                translate_coordinates(mol, transform['trans'], inverse=True)
            
        if mol is not None and len(Chem.DetectChemistryProblems(mol)) > 0:
            mol = None
        return mol


    def generate_samples(self, n_samples, temps, zbetas, input_data_file, pdb_id, savedir, ffopt=None, molfilter=None):
        if not self.model_init:
            self.initialize()
        if ffopt is None:
            ffopt = self.ffopt

        saveroot = self.get_save_path(savedir, zbetas, temps, n_samples)

        example_files = pd.read_csv(input_data_file, delim_whitespace=True, header=None, names=['recpath','ligpath'])

        n_tries_all = 0
        n_samples_all = 0

        for j in range(len(example_files)):
            recpath = example_files.loc[j, 'recpath'].replace('\\','/')
            ligpath = example_files.loc[j, 'ligpath'].replace('\\','/')

            rdlig = MolParser(ligpath).get_rdmol(sanitize=False)
            lig_center = get_mol_center(rdlig)

            savedir2 = join(saveroot, pdb_id)
            os.makedirs(savedir2, exist_ok=True)
            
            if not (self.random_rot or self.random_trans):
                with torch.no_grad() and autocast():
                    ddict, logits, output, idx1, transform, latents = self.model.get_example_from_file(ligfile=ligpath, protfile=recpath, random_rot=self.random_rot, random_trans=self.random_trans, return_latents=True)
                    zlist = latents['z']
            n_tries = 0
            mols_pred = []
            print(f'Generating samples for input {j+1:d} of {len(example_files):d}:')
            print('    recpath:', recpath)
            print('    ligpath:', ligpath)
            pbar = trange(n_samples)
            for i in pbar:
                q = 0
                mol_pred = None  
                while mol_pred is None:
                    pbar.set_description(f'({q})')
                    n_tries += 1
                    q += 1
                    with torch.no_grad() and autocast():
                        if self.random_rot or self.random_trans:
                            ddict, logits, output, idx1, transform, latents = self.model.get_example_from_file(ligfile=ligpath, protfile=recpath, random_rot=self.random_rot, random_trans=self.random_trans, return_latents=True)
                            zlist = latents['z']
                        if zbetas is not None:
                            logits = self.model.hvae_model.sample_near(ddict, num_samples=1, temps=temps, plist=None, zlist=zlist, zbeta=zbetas)
                        else:
                            logits = self.model.hvae_model.sample(ddict, num_samples=1, temps=temps)

                    mol_pred, coords_pred, types_pred, (res1_grid, res2_grid) = self.molfitter.fit_logits(logits, self.model, version='v2', set_props=True)
                    mol_pred = self.post_process_mol(mol_pred, coords_pred, molfilter, transform)
                    

                mols_pred.append(mol_pred)

            n_tries_all += n_tries
            n_samples_all += len(mols_pred)

            [mol.UpdatePropertyCache(strict=False) for mol in mols_pred]
            [Chem.GetSSSR(mol) for mol in mols_pred]            

            ###### SAVE
            # reference ligand
            lig_center = rdlig.GetConformer(0).GetPositions().mean(axis=0)
            write_mols_sdf(mols=[rdlig], file=os.path.join(savedir2, 'lig_ref.sdf'), append=True)

            # generated mols
            mols_to_write = [Chem.RemoveHs(mol, updateExplicitCount=True, sanitize=False) for mol in mols_pred]
            mols_to_write = [shift_mol(x, lig_center) for x in mols_to_write]
            molfile = os.path.join(savedir2,f'mols_gen.sdf')
            write_mols_sdf(mols_to_write, file=molfile, append=True)

        print('n_samples_all', n_samples_all)
        print('n_tries_all', n_tries_all)
        print('\n# failed: %d'%(n_tries_all-n_samples_all))
        print('\n%% failed: %.1f'%((1-n_samples_all/n_tries_all)*100))
        with open(join(saveroot,'gen.txt'), 'a+') as f:
            f.write('\n# failed: %d'%(n_tries_all-n_samples_all))
            f.write('\n%% failed: %.1f'%((1-n_samples_all/n_tries_all)*100))

        if ffopt:
            print('\n\nFF Optimizing molecules...', end=' ')
            opt_script = os.path.abspath('ff_optimize.py')
            cmd = f'python {opt_script} -d {savedir2} --overwrite --yes'
            subprocess.run(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
            print('done.')


class MolGeneratorSpatial(MolGenerator):
    '''Class for generating molecules with substructure modification.'''

    def generate_samples(self, ligfile, protfile, savedir, pdb_id, n_samples, temps, zbetas, substruct_file=None, substruct_pattern=None, ffopt=True, molfilter=None):
        if not self.model_init:
            self.initialize()
        if ffopt is None:
            ffopt = self.ffopt        

        assert (substruct_pattern is not None) != (substruct_file is not None), 'Recieved multiple inputs for substruct to modify. Input only one of `substruct_file` or `substruct_pattern`.'        
        if substruct_file is not None:
            pat_mod = MolParser(substruct_file).get_rdmol(sanitize=False)
        elif substruct_pattern is not None:
            pat_mod = Chem.MolFromSmarts(substruct_pattern)

        rdlig = MolParser(ligfile).get_rdmol(sanitize=False)
        Chem.GetSSSR(rdlig)
        
        # get substructure to modify from rdlig
        atom_idxs_mod = rdlig.GetSubstructMatches(pat_mod, useChirality=False)
        mol_mod = Chem.RWMol(rdlig)
        for i in reversed(range(rdlig.GetNumAtoms())):
            if not any([i in f for f in atom_idxs_mod]):
                mol_mod.RemoveAtom(i)
        mol_mod = Chem.Mol(mol_mod)

        lig_center = get_mol_center(rdlig)
        rdlig_centered = shift_mol(rdlig, -lig_center)
        mol_mod_centered = shift_mol(mol_mod, -lig_center)

        mol_mod_list=[mol_mod_centered]
        box_lims_all, transforms_all = self.get_box_lims(rdlig_centered, frag_mod_list=mol_mod_list, smi_mod_list=None, pad=1/24)

        ## generate
        for idx_gen in trange(len(box_lims_all)):
            frag = copy(mol_mod_list[idx_gen])
            transform_box  = transforms_all[idx_gen]

            # get ddict and latents
            ddict, _, _, _, _, latents = self.model.get_example_from_file(ligfile,
                                                                        protfile,
                                                                        transform={'matrix': transform_box['trans'], 'matrix_inv': transform_box['inv_trans']},                                                                               
                                                                        return_latents=True)
            mols_gen = []
            n_tries = 0
            for i in trange(n_samples):
                mol_pred = None
                while mol_pred is None:
                    n_tries += 1
                    with autocast():
                        logits = self.model.hvae_model.sample_spatial(ddict, num_samples=1, temps=temps, zlist=latents['z'], zbeta=zbetas, spatial_idxs=box_lims_all[idx_gen].T)

                    mol_pred, coords_pred, types_pred, _ = self.molfitter.fit_logits(logits, self.model, version='v2', set_props=True)
                    mol_pred = self.post_process_mol(mol_pred, coords_pred, molfilter, transform=None)

                transform_coordinates(mol_pred, transform_box['inv_trans'])
                mols_gen.append(mol_pred)

            n_failed = n_tries-n_samples
            print('\n# failed:', n_failed)
            print('\npct failed: %.2f'%((1-n_samples/n_tries)))

            sdir = join(savedir, f'frag_{idx_gen}', pdb_id)
            os.makedirs(sdir, exist_ok=True)
            with open(join(sdir, 'gen.txt'), 'w+') as f:
                f.write(f'ligfile: {ligfile}\n')
                f.write(f'protfile: {protfile}\n')
                f.write(f'mod_smi: {Chem.MolToSmiles(frag)}\n')
                f.write(f'mod_smarts: {Chem.MolToSmarts(frag)}\n')
                f.write(f'n_samples: {n_samples}\n')
                f.write(f'zbetas: {zbetas}\n')
                f.write(f'temps: {temps}\n')
                f.write('\n# failed: %d'%(n_failed))
                f.write('\n%% failed: %.1f'%((1-n_samples/n_tries)*100))
            
            mol_keep  = Chem.rdmolops.DeleteSubstructs(rdlig, frag)
            mols_gen_recentered = [shift_mol(m, lig_center) for m in mols_gen]
            write_mols_sdf(mols_gen_recentered, join(sdir, 'mols_gen.sdf'))
            write_mols_sdf(rdlig, join(sdir, 'lig_ref.sdf'))
            write_mols_sdf(mol_mod, join(sdir, 'mol_mod.sdf'))
            write_mols_sdf(mol_keep, join(sdir, 'mol_keep.sdf'))
            
            if ffopt:
                print('\n\nFF Optimizing molecules...', end=' ')
                opt_script = os.path.abspath('ff_optimize.py')
                cmd = f'python {opt_script} -d {os.path.abspath(sdir)} -lp *mols_gen.sdf --overwrite --yes'
                subprocess.run(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
                print('done.')
            return mols_gen_recentered

    def get_box_lims(self, mol, frag_mod_list=None, smi_mod_list=None, pad=0):
        mol0 = copy(mol)
        
        box_lims_all = []
        transforms_all = []
        
        if frag_mod_list is None:
            assert smi_mod_list is not None, 'Must input either frag_mod_list or smi_mod_list'
            frag_mod_list = [Chem.MolFromSmarts(smi) for smi in smi_mod_list]
            
        for mi, frag in enumerate(frag_mod_list):
            mol = copy(mol0)
            smarts_mod = Chem.MolToSmarts(frag)
            smi_mod = Chem.MolToSmiles(frag)
            print(f'Fragment {mi}.  Modifying: "{smi_mod}" ; "{smarts_mod}"')

            atom_idxs_mod_frags = mol.GetSubstructMatches(frag, useChirality=False)

            mol_mod = Chem.RWMol(mol)
            for i in reversed(range(mol.GetNumAtoms())):
                if not any([i in f for f in atom_idxs_mod_frags]):
                    mol_mod.RemoveAtom(i)
            mol_mod = Chem.Mol(mol_mod)

            mol_keep  = Chem.rdmolops.DeleteSubstructs(mol, frag)
            mol_keep_frags = Chem.rdmolops.GetMolFrags(mol_keep, asMols=True, sanitizeFrags=False)

            coords_mod = mol_mod.GetConformer().GetPositions()
            coords_keep = mol_keep.GetConformer().GetPositions()
            coords_keep_frags = [m.GetConformer().GetPositions() for m in mol_keep_frags]

            coords_aligned, trans, itrans = self.align_coords_box(coords_mod)
            transforms_all.append({'trans': trans, 'inv_trans':itrans})

            if trans is not None:
                coords_mod = coords_mod @ trans
                coords_keep = coords_keep @ trans

                transform_coordinates(mol_keep, trans)
                transform_coordinates(mol_mod, trans)

            box_lims = self.get_box_auto(coords_mod, coords_keep, grid_width=24, min_pad=0.02)
            box_lims = (box_lims + 0.5)
            box_lims_all.append(box_lims)
        return box_lims_all, transforms_all
    

    def align_coords_box(self, coords, coords_ref=None):
        coords_p = coords.reshape(-1,3)
        if len(coords_p) == 1:
            return coords, None, None
        if len(coords) == 2:
            n_added = 1
            coords_p = np.concatenate([coords_p, ((coords_p[1] - coords_p[0] ) + coords_p[1]).reshape(-1,3)])
        pca = PCA(n_components=3)
        if coords_ref is None:
            coords_ref = coords_p
        pca.fit(coords_p - coords_p.mean(axis=0))
        components = pca.components_
        transform = np.linalg.inv(components)
        inv_transform = components
        
        coords_aligned = coords @ transform
        
        return coords_aligned, transform, inv_transform

    
    def get_box_auto(self, coords_in, coords_out, grid_width=24, prot_density=None, min_pad=1., verbose=False):
        box_bottom = -grid_width/2
        box_top = grid_width/2
        if not isinstance(coords_out, (list,tuple)):
            coords_out = [coords_out]
        
        coords_out = [x for x in coords_out]
        
        cin = coords_in.reshape(-1,3)
        box_lims = np.empty((2,3))
        box_lims[:] = np.nan
        
        cin_lims = np.stack([cin.min(axis=0), cin.max(axis=0)]).round().astype(int)   
        cout = np.concatenate(coords_out)  
        
        bounding_planes_bottom = [[box_bottom] for _ in range(3)]
        bounding_planes_top = [[box_top] for _ in range(3)]
        for i in range(3):
            cout_vals = cout[:,i]
            
            # consider only unique values in out-coords
            bounding_planes_top[i].extend(np.unique(cout_vals[cout_vals > cin_lims[1,i]]).tolist()) 
            bounding_planes_bottom[i].extend(np.unique(cout_vals[cout_vals < cin_lims[0,i]]).tolist())
        
        # enumerate all possible bottom bounds
        bottom_bounds_all = []
        for x in bounding_planes_bottom[0]:
            for y in bounding_planes_bottom[1]:
                for z in bounding_planes_bottom[2]:
                    bottom_bounds_all.append([x,y,z])
        bottom_bounds_all = np.array(bottom_bounds_all)
                
        # enumerate all possible top bounds
        top_bounds_all = []
        for x in bounding_planes_top[0]:
            for y in bounding_planes_top[1]:
                for z in bounding_planes_top[2]:
                    top_bounds_all.append([x,y,z])
        top_bounds_all = np.array(top_bounds_all)
        
        if verbose:
            print('bottom_bounds_all:',bottom_bounds_all)
            print('top_bounds_all:',top_bounds_all)
        
        # choose bounds with largest volume that segregate coords
        bounds_best = None
        volume_best = 0
        for btop in top_bounds_all:
            for bbot in bottom_bounds_all:
                if (bbot >= btop).any():
                    continue
                coords_out_excluded = not ((cout > bbot) & (cout < btop)).all(axis=1).any()
                coords_in_included = (((cin - min_pad) > bbot) & ((cin + min_pad) < btop)).all(axis=1).all()
                if coords_out_excluded and coords_in_included:  # no out-coords within box
                    if prot_density is not None:
                        box_lims = (np.stack([bbot, btop]) / grid_width + 0.5) * prot_density.shape[-1]
                        box_lims = box_lims.round().astype(int)
                        box_vol = (prot_density == 0)[box_lims[0,0]:box_lims[1,0], box_lims[0,1]:box_lims[1,1], box_lims[0,2]:box_lims[1,2]].sum()
                    else:
                        box_vol = np.prod((btop - bbot))
                    
                    if box_vol > volume_best:
                        volume_best = box_vol
                        bounds_best = np.stack([bbot, btop]) 
        if verbose:
            print('bounds_best:', bounds_best)
        box_lims = bounds_best      
        
        return box_lims/grid_width
