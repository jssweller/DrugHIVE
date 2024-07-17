
import os, sys, time, glob, warnings
from os.path import join, basename, dirname
from tqdm import trange
import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from drughive.molecules import MolParser

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to extract. Could be single file or directory.')
    parser.add_argument('--clear', action='store_true')

    args = parser.parse_args()
    data_path = args.path
    
    grid_width = 24 # angstroms
    pad = 5 # angstroms
    dist_thresh = (2**0.5)*(grid_width/2 + pad)

    folders = [x for x in os.listdir(data_path) if not any([p in x for p in ['readme','index', '.txt']])]

    if args.clear:
        fs_remove = glob.glob(join(data_path, '**', '*.npy'), recursive=True)  ## TODO: delete
        for f in fs_remove:
            os.remove(f)
    else:
        pocket_only = True
        for i in trange(len(folders)):
            f = folders[i]

            # load ligand
            m = glob.glob(os.path.join(os.path.join(data_path,f),'*_ligand.mol2')) # ligand
            if len(m) == 0:
                continue
            mfile = m[0]

            pdb_id = basename(dirname(mfile))

            try:
                lig_parse = MolParser(fname=mfile)
                rdlig = lig_parse.get_rdmol(sanitize=False, removeHs=False)
                rdlig.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(rdlig, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY, catchErrors=True)

                atom_types_lig = lig_parse.get_atom_types(rdlig)
                atom_types_lig = lig_parse.types_list2vec(atom_types_lig)
                atom_coords_lig = rdlig.get_coords()

                idxs = atom_types_lig[:,0] != 1
                atom_types_lig = atom_types_lig[idxs]
                atom_coords_lig = atom_coords_lig[idxs]

                center_lig = atom_coords_lig.mean(axis=0)
                atom_coords_lig -= center_lig
                
                np.save(os.path.join(mfile.replace('.mol2','.coords.npy')), arr=atom_coords_lig.astype(np.float16))
                np.save(os.path.join(mfile.replace('.mol2','.types.npy')), arr=atom_types_lig.astype(np.int16))

                #load protein
                if pocket_only:
                    m = glob.glob(os.path.join(os.path.join(data_path,f),'*_pocket.pdb')) # protein pocket only
                else:
                    m = glob.glob(os.path.join(os.path.join(data_path,f),'*_protein.pdb')) # full protein
                
                if len(m) == 0:
                    continue
                mfile = m[0]

                prot_parse = MolParser(fname=mfile)
                rdprot = prot_parse.get_rdmol(remove_hetero=True, removeHs=False)
                rdprot.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(rdprot, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY, catchErrors=True)

                atom_types_prot = prot_parse.get_atom_types(rdprot)
                atom_types_prot = lig_parse.types_list2vec(atom_types_prot)
                
                atom_coords_prot = rdprot.get_coords()
                atom_coords_prot -= center_lig

                
                idxs = atom_types_prot[:,0] != 1
                atom_types_prot = atom_types_prot[idxs]
                atom_coords_prot = atom_coords_prot[idxs]

                cnorm = np.linalg.norm(atom_coords_prot, axis=-1)


                atom_coords_prot = atom_coords_prot[cnorm < dist_thresh]
                atom_types_prot = atom_types_prot[cnorm < dist_thresh]

                np.save(os.path.join(mfile.replace('.pdb','.coords.npy')), arr=atom_coords_prot.astype(np.float16))
                np.save(os.path.join(mfile.replace('.pdb','.types.npy')), arr=atom_types_prot.astype(np.int16))

            except Exception as e:
                warnings.warn(f'PDB ID: {pdb_id} failed')
                raise e
        