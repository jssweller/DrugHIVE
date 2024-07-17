#!/usr/bin/python3

from openbabel import pybel
import numpy as np
import pandas as pd
import os, sys, glob
from os.path import join, dirname, basename
from copy import copy
from tqdm import tqdm, trange
from pathlib import Path
import subprocess
import argparse
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase, RDLogger
RDLogger.DisableLog('rdApp.*')  # disable rdkit logs

from drughive.molecules import BulkSDMolParser, MolFilter, MolParser, write_mols_sdf
from drughive.trainutils import Hparams
from drughive.generating import MolGenerator, MolGeneratorSpatial


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=Path, help='Config file for generating ligands.')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', help = 'print output for each molecule')

    model_init = False
    pargs = parser.parse_args()

    if not pargs.verbose:
        blocker = rdBase.BlockLogs()
        ob_log_handler = pybel.ob.OBMessageHandler()
        ob_log_handler.SetOutputLevel(0)
        pybel.ob.obErrorLog.SetOutputLevel(0)
        
    gargs = Hparams()
    gargs.load_yaml(pargs.config_file)
    model_id0 = gargs.model_id

    molfilter = MolFilter(ring_sizes=gargs.get('ring_sizes', None), 
                          ring_system_max=gargs.get('ring_system_max', None), 
                          ring_loops_max=gargs.get('ring_loops_max', None), 
                          double_bond_pairs=gargs.get('dbl_bond_pairs', None),
                          natoms_min=gargs.get('n_atoms_min', None))
    
    print('Generating molecules and saving to:', gargs.output, flush=True)
    mod_path = gargs.get('substruct_modify_path', None)
    mod_pattern = gargs.get('substruct_modify_pattern', None)
    if mod_path or mod_pattern:
        # spatial generation mode
        assert mod_path != mod_pattern, 'Recieved multiple inputs for substructure to modify. Input only one of `path` or `pattern`.'
        molgen = MolGeneratorSpatial(gargs.checkpoint, gargs.model_id, random_rot=gargs.random_rotate, random_trans=gargs.random_translate)
        molgen.generate_samples(n_samples=gargs.n_samples, 
                                temps=gargs.get('temps', 1.),
                                zbetas=gargs.get('zbetas', 1.), 
                                ligfile=gargs.ligand_path,
                                protfile=gargs.target_path,
                                substruct_file=mod_path,
                                substruct_pattern=mod_pattern,
                                pdb_id=gargs.pdb_id,
                                savedir=gargs.output,
                                molfilter=molfilter,
                                ffopt=gargs.get('ffopt_mols', True)
                                )
    else:
        # standard generation mode
        initial_input_file = join(gargs.output, 'input.txt')
        os.makedirs(gargs.output, exist_ok=True)
        with open(initial_input_file, 'w+') as f:
            f.write(gargs['target_path'] + ' ' + gargs['ligand_path'])  
        molgen = MolGenerator(gargs.checkpoint, gargs.model_id, random_rot=gargs.random_rotate, random_trans=gargs.random_translate)
        molgen.generate_samples(gargs.n_samples, 
                            temps=gargs.get('temps', 1.),
                            zbetas=gargs.get('zbetas', 1.), 
                            input_data_file=initial_input_file,
                            pdb_id=gargs.pdb_id,
                            savedir=gargs.output,
                            molfilter=molfilter,
                            ffopt=gargs.get('ffopt_mols', True)
                            )
    