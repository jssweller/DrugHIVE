import os, glob
from collections import Counter, defaultdict
from copy import copy
import h5py as h5
import numpy as np
import warnings
from tqdm import tqdm
from rdkit import Chem

import torch
from torch.utils.data import Dataset

from .atomgridding import GridEncoder
from .molecules import AtomTyperDefault, MolParser, Mol2, removeHeteroAtoms, shift_mol
from .blocks import GaussConvLayer
from .gridutils import rot3d, trans3d, rot3d_random_safe, trans3d_random_safe


def pdb_lig_prot_from_dir(prot_dir, lig_pattern='*ligand.sdf', prot_pattern='*pocket*.pdb', remove_hs=True, recenter=True, verbose=False):
    '''Loads ligand and protein from a directory.'''
    root = prot_dir
    if verbose:
        print(os.listdir(root))
        

    protfile = glob.glob(os.path.join(root, prot_pattern))
    if len(protfile) == 0 and 'pocket' in prot_pattern:
        protfile = glob.glob(os.path.join(root,prot_pattern.replace('pocket','protein')))
    
    assert len(protfile) > 0, f'Protein file not found using glob pattern: {os.path.join(root, prot_pattern)}'
    if verbose:
        print('\nprotfile:', protfile)
    
    ligfile = glob.glob(os.path.join(root,lig_pattern))
    if verbose:
        print(ligfile)
        
    try:
        ligfile = ligfile[0]
        protfile = protfile[0]
    except Exception as e:
        print(ligfile, protfile)
        raise e
        
    return pdb_lig_prot_from_files(ligfile, protfile, remove_hs, recenter, verbose)
    
def pdb_lig_prot_from_files(ligfile, protfile, remove_hs=True, recenter=True, verbose=False):
    '''Loads ligand and protein from file paths.'''
    prot_code = os.path.basename(protfile).strip().split('_')[0]
    if verbose:
        print('prot_code: ', prot_code)
        print('protfile:', protfile)
    rdprot = Chem.rdmolfiles.MolFromPDBFile(protfile, removeHs=False, sanitize=False)
    rdprot = removeHeteroAtoms(rdprot) # remove HETATM atoms from pdb file
    
    [a.SetNoImplicit(True) for a in rdprot.GetAtoms()]
    if remove_hs:
        rdprot = Chem.RemoveHs(rdprot, sanitize=False, updateExplicitCount=True)
    Chem.GetSSSR(rdprot)

    rdprot = Mol2(rdprot)


    if ligfile.endswith('.mol2'):
        rdlig = Chem.rdmolfiles.MolFromMol2File(ligfile)
        if rdlig is None:
            rdlig = Chem.rdmolfiles.MolFromMol2File(ligfile, sanitize=False, removeHs=False, cleanupSubstructures=False)
    elif ligfile.endswith('.sdf'):
        rdlig = next(Chem.rdmolfiles.SDMolSupplier(ligfile, removeHs=False, sanitize=False))
    [a.SetNoImplicit(True) for a in rdlig.GetAtoms()]
    if remove_hs:
        rdlig = Chem.RemoveHs(rdlig, sanitize=False, updateExplicitCount=True)    
    Chem.GetSSSR(rdlig)
        
    if recenter:
        coords = rdlig.GetConformer().GetPositions()
        lig_center = coords.mean(axis=0)
        rdlig = shift_mol(rdlig, -lig_center)
        rdprot = shift_mol(rdprot, -lig_center)
    
        if verbose:
            print('lig_center:', lig_center)
    return rdlig, rdprot, prot_code


def get_pdbid_index(model, pdb_id):
    '''Gets the index of the pdb id in the model dataset.'''
    for i,f in enumerate(model.dataset.dat_files['lig_types']):
        pdb_f = os.path.basename(os.path.dirname(f)).lower() 
        if pdb_f == pdb_id.lower():
            return i
    return None    
    

def get_pdb_lig_prot(idx, model, lig_pattern='*ligand.sdf', prot_pattern='*pocket.pdb', remove_hs=True, recenter=True, verbose=False):
    '''Gets the index of the pdb id in the model dataset.'''
    prot_dir = os.path.dirname(model.dataset.dat_files['lig_types'][idx]).replace('\\', '/')
    return pdb_lig_prot_from_dir(prot_dir, lig_pattern, prot_pattern, remove_hs, recenter, verbose)


def load_dataset_rdfiles(dataset, data_root='data'):
    '''Gets paths to pdb folders for PDBBind datasets.'''
    datasets = ['pdb_refined', 'pdb_general_minus_refined', 'pdb_general', 'pdb_all']
    assert dataset in datasets, f'Invalid input. Input `dataset` must be one of {datasets}. Received "{dataset}".'
    
    if 'pdb' in dataset:
        if dataset == 'pdb_general' or dataset == 'pdb_all':
            dataset = ['pdb_general_minus_refined', 'pdb_refined']
        else:
            dataset = [dataset]
            
        folders = []
        if 'pdb_refined' in dataset:
            data_path = os.path.join(data_root, 'PDBbind_v2020_refined_all', 'refined-set')
            folders.extend([os.path.join(data_path, x) for x in os.listdir(data_path) if not any([p in x for p in ['readme','index','.txt']])])
        if 'pdb_general_minus_refined' in dataset:
            data_path = os.path.join(data_root, 'PDBbind_v2020_general_minus_refined', 'v2020-other-PL')
            folders.extend([os.path.join(data_path, x) for x in os.listdir(data_path) if not any([p in x for p in ['readme','index','.txt']])])
        print(f'Found data in {len(folders)} folders')
    return folders
        

def load_dataset_rdmols(dataset, data_root='data', pocket_only=False, ligands_only=False, num_load=None, verbose=False):
    '''Loads ligand molecules from PDBBind datasets.'''
    datasets = ['pdb_refined', 'pdb_general_minus_refined', 'pdb_general', 'pdb_all']
    assert dataset in datasets, f'Invalid input. Input `dataset` must be one of {datasets}. Received "{dataset}".'
    
    if 'pdb' in dataset:
        folders = load_dataset_rdfiles(dataset, data_root='data')
        print(f'Loading data from {len(folders)} folders')
        rdligs = []
        rdprots = []
        
        if num_load is None or num_load == 'all':
            num_load = len(folders)
        for i in tqdm(range(num_load)):
            f = folders[i]
            if (i+1) % 500 == 0 and verbose:
                print(i+1)
            molfiles = os.listdir(f)

            # load ligand
            m = [x for x in molfiles if x.endswith('.sdf')][0]
            mfile = os.path.join(f,m)
            lig_parse = MolParser(fname=mfile)
            rdlig = lig_parse.get_rdmol(sanitize=False, removeHs=True)
            rdligs.append(rdlig)
            
            if not ligands_only:
                #load protein
                if pocket_only:
                    if len([x for x in molfiles if x.endswith('pocket.pdb')]) == 0:
                        print('no pocket file', f)
                        continue
                    else:
                        m = [x for x in molfiles if x.endswith('pocket.pdb')][0] # protein pocket only
                else:

                    m = [x for x in molfiles if x.endswith('protein.pdb')][0] # protein
                mfile = os.path.join(f,m)
                prot_parse = MolParser(fname=mfile)
                rdprot = prot_parse.get_rdmol(remove_hetero=True, removeHs=True)
                rdprots.append(rdprot)
        
        if not ligands_only:
            return rdligs, rdprots, folders
        else:
            return rdligs, folders
        

def load_pdb_ligand(dir):
    """Loads ligand molecule from given directory path."""
    pattern = os.path.join(dir, '*.sdf') # ligand
    mfile = glob.glob(pattern)[0] 
    lig_parse = MolParser(fname=mfile)
    rdlig = lig_parse.get_rdmol(sanitize=False, removeHs=True)
    return rdlig

def load_pdb_protein(dir, pocket_only=False):
    """Loads protein molecule from given directory path."""
    if pocket_only:       
        pattern = os.path.join(dir, '*pocket.pdb') # protein pocket only
    else:
        pattern = os.path.join(dir, '*protein.pdb') # full protein
    mfile = glob.glob(pattern)[0]
    prot_parse = MolParser(fname=mfile)
    rdprot = prot_parse.get_rdmol(remove_hetero=True, removeHs=True)
    return rdprot

def get_pdb_folder(pdb_id, dirs=['data/PDBbind_v2020_refined_all','data/PDBbind_v2020_general_minus_refined']):
    '''Finds directory path for a pdb id.'''
    fs = []
    for d in dirs:
        fs.extend(glob.glob(os.path.join(d, '**', pdb_id), recursive=True))
    return fs

def get_natoms(atom_types, channels):
    '''Gets number of atoms in each channel.'''
    natoms = torch.zeros((len(atom_types), len(channels)), dtype=float)
    for i in range(len(atom_types)):
        for ti, t in enumerate(channels):
            c = Counter(atom_types[i])
            natoms[i,ti] = c[t]
    return natoms

def load_data_files_dict(data_paths, filter=True, filter_file=None, split='train'):
    '''Collects all data files in data_paths into a dictionary organized by type.'''
    dat_files = defaultdict(list)

    if not isinstance(data_paths, list):
        data_paths = [data_paths]

    for path in data_paths:
        folders = [x for x in os.listdir(path) if not any([p in x for p in ['readme','index']])]
        
        if filter:
            if filter_file is None:
                filter_file = 'filter_names_keep.txt'
                filter_files = glob.glob(os.path.join(path, filter_file))
                assert len(filter_files) != 0, 'Filter file not found!'
                filter_file = filter_files[0]
            filter_names = np.loadtxt(filter_file, dtype=str)
            folders = [x for x in os.listdir(path) if x in filter_names]

        patterns = {'lig_types': '*_ligand.types.npy', 
                    'lig_coords': '*_ligand.coords.npy', 
                    'prot_types': '*_pocket.types.npy', 
                    'prot_coords': '*_pocket.coords.npy'
                    }

        for i,f in enumerate(folders):
            nextfiles = {k: glob.glob(os.path.join(path,f,p)) for k,p in patterns.items()}
            if not all([len(x) == 1 for x in nextfiles.values()]):
                print('Files Not Found... skipping folder: %s'%f)
                continue
            
            for k,v in nextfiles.items():
                dat_files[k].append(v[0])
    return dat_files


class MolDataset(Dataset):
    """Molecular dataset."""
    TYPES = [6, 7, 8, 9, 16, 17, 35, 53]

    def __init__(self, data_path, params, idxs=None, multi_read=False, split='train'):
        """
        Args:
            data_path (string): Path to dataset (hdf5 format).
            params: dictionary of parameters
        """
        
        self.coords_data = None
        self.types_data = None
        self.dims = None
        self.split = 'split'

        self.idxs = idxs
        
        self.params = params
        self.data_path = data_path
        self.noise_encoding = params.get('noise_encoding', {0: False})
        self.noise_addatom = params.get('noise_addatom', {0: False})
        self.noise_removeatom = params.get('noise_removeatom', False)
        self.random_rot = params.get('random_rotate', False)
        self.random_trans = params.get('random_translate', False)
        self.random_types = params.get('random_types', False)
        self.random_channels = params.get('random_channels', False)
        self.trans_max_dist = params.get('translate_max_dist', False)
        self.encoding = params.get('encoding', 'soft')
        self.normalize = params.get('normalize', False)

        self.norm_w = 1.
        if self.encoding in ['soft3', 'soft']:
            self.norm_w = 3.
                
        self.grid_size = params.get('grid_size', 19.5)
        self.resolution = params.get('resolution', 0.4)
        self.n_batch = params.get('n_batch', 1)
        self.device = params.get('device', 'cpu')
        self.multi_read = multi_read            
        self.atomtyper = AtomTyperDefault()
        
        
        channels_default = params.get('channels', MolDataset.TYPES)
        self.channels_in = params.get('channels_in') if params.get('channels_in') else channels_default
        self.channels_out = params.get('channels_out') if params.get('channels_out') else self.channels_in
        
        self.grid_encoder = GridEncoder(grid_size=self.grid_size,
                                resolution=self.resolution, 
                                n_batch=self.n_batch, 
                                channels=self.channels_in)

        self.channels_in = copy(self.grid_encoder.channels)
        self.grid_encoder.channels = self.channels_out
        self.channels_out = copy(self.grid_encoder.channels)
        
        self.channels = copy(self.channels_in)
        for ch in self.channels_out:
            if ch not in self.channels:
                self.channels.append(ch)
        
        self.grid_encoder.channels = self.channels
        self.grid_encoder.init_grid()
        

        self.channels_in_idxs = [self.channels.index(x) for x in self.channels_in if x in self.channels]
        self.channels_out_idxs = [self.channels.index(x) for x in self.channels_out if x in self.channels]
        
        
        if self.noise_addatom['0']:
            self.atom_noise_encoder = GridEncoder(grid_size=self.grid_size, 
                                            resolution=self.resolution, 
                                            n_batch=self.n_batch,
                                            channels=self.channels_in)

    def __len__(self):
        self.init_check()
        return len(self.idxs)

    def __getitem__(self, idx, verbose=False):
        self.init_check()
        
        if isinstance(idx,int):
            idx = [self.idxs[idx]]
        elif isinstance(idx, slice):
            idx = self.idxs[idx].tolist()
        elif torch.is_tensor(idx):
            idx = self.idxs[idx].tolist()
        n_batch = len(idx)
        
        # create new grid_encoder each time to prevent issues parallel data loading (num_workers > 1)
        grid_encoder = GridEncoder(grid_size=self.grid_size, 
                                        resolution=self.resolution, 
                                        n_batch=self.n_batch, 
                                        channels=self.channels)

        if grid_encoder.n_batch < n_batch:
            grid_encoder.n_batch = n_batch

        grid_encoder.init_grid()
        
        sample_dict = {}
        sample_dict['transform'] = {}
        atom_coords = self.coords_data[idx]
        atom_types = self.types_data[idx]

        if self.random_rot:
            atom_coords, rot_angles, in_bounds = rot3d_random_safe(atom_coords, grid_encoder, pad=1.5, n_attempts=10, voxels=False)
            sample_dict['transform']['rotate'] = rot_angles
        if self.random_trans:
            atom_coords, trans_vec, in_bounds = trans3d_random_safe(atom_coords, grid_encoder, max_dist=self.trans_max_dist, pad=1.5, n_attempts=10, voxels=False)
            sample_dict['transform']['translate'] = trans_vec
        
        
        grid_encoder.atom_coords = atom_coords
        grid_encoder.atom_types = atom_types
        
        atom_coords = grid_encoder.atom_coords
        atom_coords0 = copy(atom_coords)

        # randomly assign element types to atoms without adding new atoms
        if self.random_types:
            egrid = grid_encoder
            types1hot = egrid.atom_types.bool()
            element_nums = np.concatenate([egrid.channel_nums[x] for x in egrid.channels_elements_only]).tolist() # all element types in channels
            element_channel_bool = grid_encoder.atom_channels_1hot[:,:,egrid.channels_elements_only].any(axis=-1)  # atoms with nonzero element type
            
            elem_types_new = np.zeros((*types1hot.shape[:2],len(element_nums)))
            for i in range(len(types1hot)):
                for j in range(len(element_nums)):
                    prob = np.random.random() * 0.7
                    elem_types_new[i,:,j] = np.random.choice([True,False], p=[prob,1-prob], size=(types1hot.shape[1]))
                # Fill empty rows with single element type to make sure molecule is full
                fill_num = np.random.randint(elem_types_new.shape[-1])
                empty_idxs = np.arange(elem_types_new.shape[1])[~elem_types_new[i].any(axis=-1)]
                elem_types_new[i,empty_idxs,fill_num] = 1

            elem_types_new *= element_channel_bool.numpy().reshape(*element_channel_bool.shape,1)
            types1hot[:,:,element_nums] = torch.from_numpy(elem_types_new.astype(bool))
            grid_encoder.atom_types = types1hot
        
        ## add bonded to channels
        if len(grid_encoder.channels_bto_only) > 0:
            grid_encoder.update_bonded_to()
        
        grid_encoder.encode_coords2grid(encoding=self.encoding)
        
        sample_dict['natoms'] = grid_encoder.atom_channels_1hot.sum(axis=1)
        if len(grid_encoder.channels_bonds_only) > 0:
            sample_dict['natoms'] += grid_encoder.bond_channels_1hot.sum(axis=1)
        sample_dict['types'] = grid_encoder.atom_types
        sample_dict['coords'] = atom_coords
        
        sample_dict['target_encoding'] = grid_encoder.values[:n_batch, self.channels_out_idxs].clone()
        sample_dict['input_encoding'] = grid_encoder.values[:n_batch, self.channels_in_idxs].clone() 

    
        if self.noise_addatom['0']:
            atom_noise_p = self.noise_addatom.get('prob',0.1) # probability of adding atom noise to a molecule
            atom_noise_mult = self.noise_addatom.get('mult',.5)  # multiplier of atom noise values. Determines relative amplitude compared to normal atom
            noise_coords = torch.empty([1,1,3])
            noise_coords[:] = torch.nan
            noise_types = torch.zeros([1,1], dtype=int)
            noise_natoms = torch.zeros([1,len(self.channels_in)], dtype=int)
            if np.random.random() < atom_noise_p:               
                noise_r = 2  # minimum distance to nearest atom
                n_atoms_noise = 1 # number of atoms to add per molecule
                noise_coords = torch.zeros((n_atoms_noise),3)
                atom_coords = torch.tensor(atom_coords[0], dtype=float)
                for i in range(n_atoms_noise):
                    newcoords = atom_coords[0]
                    while torch.norm(torch.cat([atom_coords[~torch.isnan(atom_coords).any(dim=1)], noise_coords], dim=0) - newcoords, dim=1).min() < noise_r:
                        newcoords = (torch.rand(1,3)-0.5)*(self.atom_noise_encoder.grid_size-noise_r)
                    noise_coords[i] = newcoords
                
                elements = np.concatenate([self.atom_noise_encoder.channels[x] for x in \
                                           self.atom_noise_encoder.channels_elements_only]).tolist()
                
                elements = [grid_encoder.atomtyper.type2num(x) for x in elements]
                
                noise_types = torch.tensor(np.random.choice(elements, 
                                                            size=len(noise_coords), 
                                                            replace=True).reshape(1,-1), dtype=int)
                    
                noise_coords = noise_coords.unsqueeze(0)
                    
                self.atom_noise_encoder.encode_coords2grid(coords=noise_coords, types=noise_types, encoding=self.encoding)
                self.atom_noise_encoder.values *= atom_noise_mult
                
                noise_natoms = self.atom_noise_encoder.atom_channels_1hot.sum(axis=1).int()

            sample_dict['noise_coords'] = noise_coords
            sample_dict['noise_types'] = noise_types
            sample_dict['noise_natoms'] = noise_natoms
            sample_dict['input_encoding'] += self.atom_noise_encoder.values
            
        sample_dict['natoms'] = sample_dict['natoms'][:n_batch, self.channels_out_idxs]
        
        if self.noise_encoding['0']:
            enc_noise_p = self.noise_encoding.get('prob', 0.05) # proportion of atoms with encoding noise added
            enc_noise_max = self.noise_encoding.get('max_val', 0.5) # max voxel value of noise
            self.enc_noise_encoder = self.atom_noise_encoder.copy()
            self.enc_noise_encoder.init_grid()
            self.enc_noise_encoder.atom_coords = grid_encoder.atom_coords
            self.enc_noise_encoder.atom_types = grid_encoder.atom_types
            self.enc_noise_encoder.add_encoding_noise(noise_p=enc_noise_p, max_val=enc_noise_max)
            sample_dict['input_encoding'] += self.enc_noise_encoder.values
        
        if n_batch == 1:  # squeeze batch dimension for pytorch Dataloader
            for key in sample_dict.keys():
                if key not in ['transform']:
                    sample_dict[key] = sample_dict[key].squeeze(0)
        
        if self.normalize and self.encoding in ['soft3','soft']:
            sample_dict['input_encoding'] /= self.norm_w
            sample_dict['target_encoding'] /= self.norm_w

        return sample_dict

    def init_check(self):
        '''Checks if data is already loaded, and if not loads the data'''
        if self.types_data is None or self.coords_data is None:
            d = self.data_path
            if os.path.isfile(d):
                d = os.path.dirname(d)
            
            self.data = h5.File(self.data_path, 'r', libver='latest', swmr=True)
            self.types_data = self.data['atom_types']
            self.coords_data = self.data['atom_coords']

            if 'zinc' in d.lower():
                if os.path.isfile(os.path.join(d, f'idxs_{self.split}.npy')):
                    self.idxs = np.load(os.path.join(d, f'idxs_{self.split}.npy')).astype(int)
                else:
                    self.idxs = np.arange(len(self.coords_data), dtype=int)


class MolDatasetPDBBind(Dataset):
    """Molecular dataset class for PDBBind data."""
    TYPES_LIG = ['e6', 'e7', 'e8', 'e9', 'e15', 'e16', 'e17', 'e35', 'e53', 'HBD', 'HBA', 'Aromatic', 'FChargePos', 'FChargeNeut', 'FChargeNeg']
    TYPES_PROT = ['e6', 'e7', 'e8', 'e16', 'HBD', 'HBA', 'Aromatic', 'FChargePos', 'FChargeNeut', 'FChargeNeg']

    def __init__(self, data_path, params, multi_read=False, split='train'):
        """
        Args:
            data_path (string): Path to dataset root directory.
            params: dictionary of parameters
        """
        
        self.coords_lig_data = None
        self.coords_prot_data = None
        self.dat_files = None
        self.dims = None

        self.filter_data = params.get('filter_data', True)
        self.split = split
        
        self.params = params
        self.data_path = data_path

        self.random_rot = params.get('random_rotate', False)
        self.random_trans = params.get('random_translate', False)
        self.random_types = params.get('random_types', False)
        self.random_channels = params.get('random_channels', False)
        self.trans_max_dist = params.get('translate_max_dist', 2)
        self.encoding = params.get('encoding', 'softcube')
        self.normalize = params.get('normalize', True)

        self.norm_w = 1

        self.grid_size = params.get('grid_size', 24)
        self.resolution = params.get('resolution', 0.5)
        self.n_batch = params.get('n_batch', 1)
        self.device = params.get('device', 'cpu')
        self.multi_read = multi_read
        self.atomtyper = AtomTyperDefault()

        
        
        self.channels_in_lig = params.get('channels_in_lig') if params.get('channels_in_lig') else MolDatasetPDBBind.TYPES_LIG
        self.channels_in_prot = params.get('channels_in_prot') if params.get('channels_in_prot') else MolDatasetPDBBind.TYPES_PROT
        self.channels_out_lig = params.get('channels_out_lig') if params.get('channels_out_lig') else self.channels_in_lig
        self.channels_out_prot = params.get('channels_out_prot') if params.get('channels_out_prot') else self.channels_in_prot
                
        self.grid_encoder_ligand = GridEncoder(grid_size=self.grid_size, 
                                resolution=self.resolution, 
                                n_batch=self.n_batch, 
                                channels=self.channels_in_lig)

                                
        self.grid_encoder_protein = GridEncoder(grid_size=self.grid_size, 
                                resolution=self.resolution, 
                                n_batch=self.n_batch, 
                                channels=self.channels_in_prot)

        self.grid_encoder = self.grid_encoder_ligand  # ligand encoder is default grid_encoder

        self.channels_in_lig = copy(self.grid_encoder_ligand.channels)
        self.grid_encoder_ligand.channels = self.channels_out_lig
        self.channels_out_lig = copy(self.grid_encoder_ligand.channels)
        
        self.channels_lig = copy(self.channels_in_lig)
        for ch in self.channels_out_lig:
            if ch not in self.channels_lig:
                self.channels_lig.append(ch)

        self.channels_in_prot = copy(self.grid_encoder_protein.channels)
        self.grid_encoder_protein.channels = self.channels_out_prot
        self.channels_out_prot = copy(self.grid_encoder_protein.channels)
        
        self.channels_prot = copy(self.channels_in_prot)
        for ch in self.channels_out_prot:
            if ch not in self.channels_prot:
                self.channels_prot.append(ch)
        
        

        self.grid_encoder_ligand.channels = self.channels_lig
        self.grid_encoder_ligand.init_grid()

        self.grid_encoder_protein.channels = self.channels_prot
        self.grid_encoder_protein.init_grid()

        self.channels_in_lig_idxs = [self.channels_lig.index(x) for x in self.channels_in_lig if x in self.channels_lig]
        self.channels_out_lig_idxs = [self.channels_lig.index(x) for x in self.channels_out_lig if x in self.channels_lig]

        self.channels_in_prot_idxs = [self.channels_prot.index(x) for x in self.channels_in_prot if x in self.channels_prot]
        self.channels_out_prot_idxs = [self.channels_prot.index(x) for x in self.channels_out_prot if x in self.channels_prot]       

    def __len__(self):
        return len(self.dat_files[list(self.dat_files.keys())[0]])

    def __getitem__(self, idx, verbose=False):
        self.init_check()
        
        if isinstance(idx,int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(0, idx.stop)[idx])
        elif torch.is_tensor(idx):
            idx = idx.tolist()
        n_batch = len(idx)

        atom_types_lig = np.zeros((n_batch, self.coords_lig_nmax, self.types_lig_nmax), dtype=int) - 1
        atom_types_prot = np.zeros((n_batch, self.coords_prot_nmax, self.types_prot_nmax), dtype=int) - 1
        atom_coords_lig = np.zeros((n_batch, self.coords_lig_nmax, 3))
        atom_coords_lig[:] = np.nan
        atom_coords_prot = np.zeros((n_batch, self.coords_prot_nmax, 3))
        atom_coords_prot[:] = np.nan

        for j, i in enumerate(idx):
            vals = self.types_lig_data[i]
            atom_types_lig[j,:len(vals), :vals.shape[-1]] = vals

            vals = self.coords_lig_data[i]
            atom_coords_lig[j,:len(vals)] = vals

            vals = self.types_prot_data[i]
            atom_types_prot[j,:len(vals), :vals.shape[-1]] = vals

            vals = self.coords_prot_data[i]
            atom_coords_prot[j,:len(vals)] = vals

        return self.batch_from_coords_types(atom_coords_lig, atom_types_lig, atom_coords_prot, atom_types_prot)


    def batch_from_file(self, ligfile, protfile):
        '''Loads an example as a batch from ligand and protein files.'''
        molparser = MolParser()

        coords_lig, types_lig = molparser.mol_data_from_file(ligfile)
        coords_prot, types_prot = molparser.mol_data_from_file(protfile)

        center_lig = coords_lig.mean(axis=0)
        coords_lig -= center_lig
        coords_prot -= center_lig

        # filter out protein atoms beyond a distance threshold
        dist_thresh = self.grid_encoder_ligand.grid_size
        prot_mask = np.linalg.norm(coords_prot, axis=-1) < dist_thresh
        coords_prot = coords_prot[prot_mask]
        types_prot = types_prot[prot_mask]

        return self.batch_from_coords_types(coords_lig, types_lig, coords_prot, types_prot)


    def batch_from_coords_types(self, atom_coords_lig, atom_types_lig, atom_coords_prot, atom_types_prot):
        '''Loads an example as a batch from atomic coordinates and types for ligand and protein.'''
        # create new grid_encoder each time to prevent issues parallel data loading (num_workers > 1)
        self.ge_prot_batch = self.grid_encoder_protein.copy()
        self.ge_lig_batch = self.grid_encoder_ligand.copy()

        # reshape if data is not batched
        if atom_coords_lig.ndim == 3:
            atom_coords_lig.reshape(1, *atom_coords_lig.shape)
        if atom_types_lig.ndim == 3:
            atom_types_lig.reshape(1, *atom_types_lig.shape)
        if atom_coords_prot.ndim == 3:
            atom_coords_prot.reshape(1, *atom_coords_prot.shape)
        if atom_types_prot.ndim == 3:
            atom_types_prot.reshape(1, *atom_types_prot.shape)
        
        n_batch = len(atom_coords_lig)

        if self.ge_prot_batch.n_batch < n_batch:
            self.ge_prot_batch.n_batch = n_batch
        if self.ge_lig_batch.n_batch < n_batch:
            self.ge_lig_batch.n_batch = n_batch

        sample_dict = {}
        sample_dict['transform'] = {}

        if self.random_rot:
            atom_coords_lig, rot_angles, success_bool = rot3d_random_safe(atom_coords_lig, grid_encoder=self.ge_lig_batch, pad=1.5, voxels=False)
            atom_coords_prot = rot3d(atom_coords_prot, rot_angles) # rotate protein by same amount
            sample_dict['transform']['rotate'] = rot_angles
        if self.random_trans:
            atom_coords_lig, trans_vec, success_bool = trans3d_random_safe(atom_coords_lig, max_dist=self.trans_max_dist, grid_encoder=self.ge_lig_batch, pad=1.5, voxels=False)
            atom_coords_prot = trans3d(atom_coords_prot, trans_vec)  # translate protein by same amount
            sample_dict['transform']['translate'] = trans_vec       
        
        self.ge_lig_batch.atom_coords = atom_coords_lig
        self.ge_lig_batch.atom_types = atom_types_lig

        self.ge_prot_batch.atom_coords = atom_coords_prot
        self.ge_prot_batch.atom_types = atom_types_prot

        atom_coords_lig = self.ge_lig_batch.atom_coords
        atom_coords_prot = self.ge_prot_batch.atom_coords
        
        ## add bonded to channels
        if len(self.ge_lig_batch.channels_bto_only) > 0:
            self.ge_lig_batch.update_bonded_to()
        if len(self.ge_prot_batch.channels_bto_only) > 0:
            self.ge_prot_batch.update_bonded_to()
        
        self.ge_lig_batch.encode_coords2grid(encoding=self.encoding)    
        self.ge_prot_batch.encode_coords2grid(encoding=self.encoding)        
        
        sample_dict['types_lig'] = self.ge_lig_batch.atom_types
        sample_dict['coords_lig'] = atom_coords_lig
        
        sample_dict['types_prot'] = self.ge_prot_batch.atom_types
        sample_dict['coords_prot'] = atom_coords_prot
        
        sample_dict['target_encoding_lig'] = self.ge_lig_batch.values[:n_batch, self.channels_out_lig_idxs].clone()
        sample_dict['input_encoding_lig'] = self.ge_lig_batch.values[:n_batch, self.channels_in_lig_idxs].clone()
        
        sample_dict['target_encoding_prot'] = self.ge_prot_batch.values[:n_batch, self.channels_out_prot_idxs].clone()
        sample_dict['input_encoding_prot'] = self.ge_prot_batch.values[:n_batch, self.channels_in_prot_idxs].clone()


        if n_batch == 1:  # squeeze batch dimension for pytorch Dataloader
            for key in sample_dict.keys():
                if key not in ['transform']:
                    sample_dict[key] = sample_dict[key].squeeze(0)


        if self.normalize and self.norm_w != 1:
            sample_dict['input_encoding_lig'] /= self.norm_w
            sample_dict['target_encoding_lig'] /= self.norm_w
            sample_dict['input_encoding_prot'] /= self.norm_w
            sample_dict['target_encoding_prot'] /= self.norm_w

            if self.split_res:
                sample_dict['input_encoding_lig2'] /= self.norm_w
                sample_dict['target_encoding_lig2'] /= self.norm_w
                sample_dict['input_encoding_prot2'] /= self.norm_w
                sample_dict['target_encoding_prot2'] /= self.norm_w
        
        return sample_dict


    def init_check(self):
        if self.coords_lig_data is None or self.coords_prot_data is None:
            if self.filter_data:
                f = glob.glob(os.path.join(self.data_path, '**',f'*{self.split}_split.csv'), recursive=True)
                f = f[0] if len(f) > 0 else None
            self.dat_files = load_data_files_dict(self.data_path, filter=self.filter_data, filter_file=f)

            self.coords_lig_data = [np.load(x) for x in self.dat_files['lig_coords']]
            self.coords_prot_data = [np.load(x) for x in self.dat_files['prot_coords']]

            self.types_lig_data = [np.load(x).astype(int) for x in self.dat_files['lig_types']]
            self.types_prot_data = [np.load(x).astype(int) for x in self.dat_files['prot_types']]

            self.coords_lig_nmax = max([len(x) for x in self.coords_lig_data])
            self.coords_prot_nmax = max([len(x) for x in self.coords_prot_data])
            self.types_lig_nmax = max([x.shape[-1] for x in self.types_lig_data])
            self.types_prot_nmax = max([x.shape[-1] for x in self.types_prot_data])


class MolDatasetPDBBindZINCSplit(Dataset):
    """Molecular dataset class for loading PDBBind and ZINC data into grids at split resolutions."""
    TYPES_LIG = ['e6', 'e7', 'e8', 'e9', 'e15', 'e16', 'e17', 'e35', 'e53', 'HBD', 'HBA', 'Aromatic', 'FChargePos', 'FChargeNeut', 'FChargeNeg']
    TYPES_PROT = ['e6', 'e7', 'e8', 'e16', 'HBD', 'HBA', 'Aromatic', 'FChargePos', 'FChargeNeut', 'FChargeNeg']

    def __init__(self, params, data_path_zinc=None, data_path_pdb=None, multi_read=False, split='train'):
        """
        Args:
            data_path_zinc (string): Path to dataset root directory.
            data_path_pdb_bind (string): Path to dataset root directory.
            params: dictionary of parameters
        """

        self.input_grid_format2 = params.get('input_grid_format2', 'density')
        self.output_grid_format2 = params.get('output_grid_format2', 'density')

        params['channels_in'] = params.get('channels_in_lig')
        params['channels_out'] = params.get('channels_out_lig')

        self.split = split
        self.params = params
        if data_path_zinc is None:
            data_path_zinc = params.get('data_path_zinc', 'data/ZINC12_clean/zinc12_clean_balanced.h5')
        if data_path_pdb is None:
            data_path_pdb = params.get('data_path_pdb', 'data/PDBbind_v2020_refined_all/refined-set')

        self.dataset_zinc = MolDataset(data_path_zinc, params=params, multi_read=multi_read, split=split)
        self.dataset_pdb = MolDatasetPDBBind(data_path_pdb, params=params, multi_read=multi_read, split=split)

        self.grid_encoder_ligand = self.dataset_pdb.grid_encoder_ligand
        self.grid_encoder = self.grid_encoder_ligand
        self.grid_encoder_protein = self.dataset_pdb.grid_encoder_protein

        self.downsample_mode = params.get('downsample_mode', 'trilinear')
        self.random_channels = params.get('random_channels', False)
            

        self.shapes = None
        self.keys_pdb = None

        self.gauss_conv_ligand = GaussConvLayer(var=self.params['gauss_var'], 
                                                trunc=self.params['gauss_trunc'], 
                                                resolution=self.params['resolution'], 
                                                n_channels=len(self.grid_encoder_protein.channels))
        
        self.gauss_conv_protein = GaussConvLayer(var=self.params['gauss_var'], 
                                                trunc=self.params['gauss_trunc'], 
                                                resolution=self.params['resolution'], 
                                                n_channels=len(self.grid_encoder_ligand.channels))


    @property
    def random_channels(self):
        return self._random_channels
    
    @random_channels.setter
    def random_channels(self, val):
        self._random_channels = val
        self.dataset_pdb.random_channels = val
        self.dataset_zinc.random_channels = val

    @property
    def cycle_zinc(self):
        return self.cycle_zinc_

    @cycle_zinc.setter
    def cycle_zinc(self, val):
        self.cycle_zinc_ = val
        self.init_zinc_counter()

    @property
    def zinc_alpha(self):
        return self.zinc_alpha_   
    
    @zinc_alpha.setter
    def zinc_alpha(self, val):
        self.zinc_alpha_ = val
        self.zinc_n = int(self.zinc_alpha * len(self.dataset_pdb)) # number of ZINC examples in each epoch
        if hasattr(self, 'zinc_counter') and len(self.zinc_counter) != self.zinc_n:
            self.reinit_zinc_counter(self.zinc_n)      
    
    @property
    def random_rot(self):
        return self.random_rot_
    
    @random_rot.setter
    def random_rot(self, val):
        self.random_rot_ = val
        self.dataset_pdb.random_rot = val
        self.dataset_zinc.random_rot = val
    
    @property
    def random_trans(self):
        return self.random_trans_
    
    @random_trans.setter
    def random_trans(self, val):
        self.random_trans_ = val
        self.dataset_pdb.random_trans = val
        self.dataset_zinc.random_trans = val

    @property
    def dat_files(self):
        return self.dataset_pdb.dat_files
    
    @property
    def channels_lig(self):
        return self.dataset_pdb.channels_lig
    
    @channels_lig.setter
    def channels_lig(self, val):
        # self.channels_lig_ = val
        self.dataset_pdb.channels_lig = val
        self.dataset_zinc.channels = val
    
    @property
    def channels_prot(self):
        return self.dataset_pdb.channels_prot
    
    @channels_prot.setter
    def channels_prot(self, val):
        # self.channels_prot_ = val
        self.dataset_pdb.channels_prot = val


    def update_idxs(self):
        self.idxs_zinc = np.random.randint(len(self.dataset_zinc), size=self.zinc_n)
    
    def __len__(self):
        return  self.zinc_n + len(self.dataset_pdb)
    
    def init_shapes(self):
        '''Gets the shapes of each data type and stores as dictionary attribute.'''
        shapes = {}
        d_pdb = self.dataset_pdb[0]
        d_zinc = self.dataset_zinc[0]

        for key in ['types_lig', 'types_prot', 'coords_lig', 'coords_prot']:
            p = d_pdb[key].shape
            if '_prot' in key:
                shapes[key] = p[0]
            else:
                z = d_zinc[key.replace('_lig','').replace('_prot','')].shape
                shapes[key] = max(z[0], p[0])

        self.shapes = shapes
        self.keys_pdb = list(d_pdb.keys())


    def init_zinc_counter(self):
        '''Keeps track of number of times each idx has been passed to __getitem__'''
        self.zinc_counter = np.zeros(self.zinc_n, dtype=int)


    def zinc_counter_idxs_visited(self, zinc_visited):
        '''Reinitializes zinc_counter with a different value of zinc_n. Copy all visited indices to new counter.'''
        idxs_v = []
        for i in range(len(zinc_visited)):
            for j in range(int(zinc_visited[i])):
                idx = i + len(zinc_visited)*j
                idxs_v.append(idx)
        return idxs_v


    def reinit_zinc_counter(self, zinc_n):
        '''Reinitializes zinc_counter and sets zinc_n (number of ZINC examples in each epoch).'''
        zc_old = self.zinc_counter
        zc_new = np.zeros(zinc_n)
        idxs_v = self.zinc_counter_idxs_visited(zc_old)
        for idx in idxs_v:
            i2 = idx % zinc_n
            zc_new[i2] += 1
        self.zinc_counter = zc_new


    def cycle_zinc_idxs(self, idxs_zinc):
        '''Converts input indices into ZINC dataset indices. Updates zinc_counter.'''
        for i, idx in enumerate(idxs_zinc):
            newidx = idx + int(self.zinc_counter[idx] * self.zinc_n)
            if newidx >= len(self.dataset_zinc):
                self.zinc_counter[idx] = 0
                newidx = idx

            idxs_zinc[i] = newidx
            self.zinc_counter[idx] += 1
        return idxs_zinc
    

    def __getitem__(self, idx, verbose=False):
        self.init_check()     
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = list(range(0, idx.stop)[idx])
        elif torch.is_tensor(idx):
            idx = idx.tolist()

        idxs_zinc = [i-len(self.dataset_pdb) for i in idx if i >= len(self.dataset_pdb)]
        idxs_pdb = [i for i in idx if i < len(self.dataset_pdb)]

        # cycle through zinc dataset
        if self.cycle_zinc:
            idxs_zinc = self.cycle_zinc_idxs(idxs_zinc)

        if self.shapes is None:
            self.init_shapes()

        sample_dict = {}
        if len(idxs_zinc) > 0:
            sample_dict_zinc = self.dataset_zinc[idxs_zinc]
            sample_dict_zinc['types_lig'] = sample_dict_zinc.pop('types')
            sample_dict_zinc['coords_lig'] = sample_dict_zinc.pop('coords')


            sample_dict_zinc['types_prot'] = torch.zeros([self.shapes['types_prot']]+list(sample_dict_zinc['types_lig'].shape[1:]), dtype=sample_dict_zinc['types_lig'].dtype)
            sample_dict_zinc['coords_prot'] = torch.zeros([self.shapes['coords_prot']]+list(sample_dict_zinc['coords_lig'].shape[1:]), dtype=sample_dict_zinc['coords_lig'].dtype)
            sample_dict_zinc['coords_prot'][:] = torch.nan
            

            sample_dict_zinc['target_encoding_lig'] = sample_dict_zinc.pop('target_encoding')
            sample_dict_zinc['input_encoding_lig'] = sample_dict_zinc.pop('input_encoding')

            sample_dict_zinc['target_encoding_prot'] = torch.zeros_like(sample_dict_zinc['target_encoding_lig'])
            sample_dict_zinc['input_encoding_prot'] = torch.zeros_like(sample_dict_zinc['input_encoding_lig'])


        if len(idxs_pdb) > 0:
            sample_dict = self.dataset_pdb[idxs_pdb]

            if len(idxs_zinc) > 0:
                sample_dict['coords_lig'] = torch.concat([sample_dict['coords_lig'], sample_dict_zinc['coords_lig']])
                sample_dict['types_lig'] = torch.concat([sample_dict['types_lig'], sample_dict_zinc['types_lig']])

                sample_dict['target_encoding_lig'] = torch.concat([sample_dict['target_encoding_lig'], sample_dict_zinc['target_encoding_lig']])
                sample_dict['input_encoding_lig'] = torch.concat([sample_dict['input_encoding_lig'], sample_dict_zinc['input_encoding_lig']])
                
                sample_dict['target_encoding_lig'] = torch.concat([sample_dict['target_encoding_prot'], sample_dict_zinc['target_encoding_prot']])
                sample_dict['input_encoding_lig'] = torch.concat([sample_dict['input_encoding_prot'], sample_dict_zinc['input_encoding_prot']])

                if self.dataset_pdb.random_rot:
                    sample_dict['transform']['rotate'] = np.concatenate([sample_dict['transform']['rotate'], sample_dict_zinc['transform']['rotate']])
                if self.dataset_pdb.random_trans:
                    sample_dict['transform']['translate'] = np.concatenate([sample_dict['transform']['translate'], sample_dict_zinc['transform']['translate']])

        else:
            sample_dict = {key:sample_dict_zinc[key] for key in self.keys_pdb}
            

        for key in ['types_lig', 'types_prot', 'coords_lig', 'coords_prot']:
            n_default = self.shapes[key]
            n = len(sample_dict[key])
            
            assert n <= n_default, 'length of %d for `%s` is invalid for default length of %d'%(len(sample_dict[key]), key, n_default)

            if n < n_default:
                fill_vals = torch.zeros(size=[n_default-n]+list(sample_dict[key].shape[1:]), dtype=sample_dict[key].dtype)
                if key.startswith('coords_'):
                    fill_vals[:] = torch.nan
                sample_dict[key] = torch.concat([sample_dict[key], fill_vals])

        return_keys = ['input_encoding_lig', 'input_encoding_prot']

        for key in return_keys:
            if 'encoding' in key or 'density' in key:
                if sample_dict[key].ndim==5 and len(sample_dict[key]) == 1:
                    sample_dict[key] = sample_dict[key].squeeze(0) 
                   
        self.sample_dict = sample_dict
        return {key: sample_dict[key] for key in return_keys}
    
    def init_check(self):
        if self.zinc_alpha is None:
            self.zinc_alpha = self.params.get('zinc_alpha', 1.)
            self.cycle_zinc = self.params.get('cycle_zinc', True)
    