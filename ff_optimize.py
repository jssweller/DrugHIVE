#!/usr/bin/python3
import glob, sys, os
import subprocess
from tqdm import trange
import pandas as pd
import time
from os.path import join, basename, dirname
import argparse

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from drughive.molecules import optimize_mols_multi, BulkSDMolParser, MolParser, write_mols_sdf, get_largest_fragment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required = False, default=None,  help = 'directory to process containing')
    parser.add_argument('-lp', '--lig_patterns', required = False, nargs='+', default=None,  help = 'list of regex patterns for finding ligand files (separated by spaces).')
    parser.add_argument('-l', '--ligand', dest = 'ligfile', required = False, default=None, help = 'input ligand file (.sdf or .sdf.gz)')
    parser.add_argument('-o', '--out', dest = 'outfile', required = False, default=None, help = 'output ligand file (.sdf or .sdf.gz)')
    parser.add_argument('-f', '--overwrite', required=False, action='store_true', help = 'Re-process files if output already exists.')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', help = 'print output for each molecule')
    parser.add_argument('-y', '--yes', required=False, action='store_true', help = 'Automatically accept prompts.')

    args = parser.parse_args()

    global LIG_PATTERNS_EXCLUDE
    LIG_PATTERNS_EXCLUDE = ['smina', '_opt', 'qvina']
    t0 = time.time()

    ligfiles_process = []

    print('Searching for ligands..', flush=True)
    if args.directory is None:
        assert args.ligfile is not None, 'Either provide `--directory` or `--ligand`.'
        ligfiles_process.append([args.ligfile])
    else:
        assert args.ligfile is None, 'Conflicting flags. Do not provide --ligand when --directory is provided.'
        if args.lig_patterns is None:
            args.lig_patterns = ['lig_ref*.sdf', 'mols_pred*.sdf', 'mols_gen*.sdf']

        # get directories to process by searching for lig_patterns in directory tree
        dirs_process = []
        [dirs_process.extend(glob.glob(os.path.join(args.directory, f'**/{pat}'), recursive=True)) for pat in args.lig_patterns]
        [dirs_process.extend(glob.glob(os.path.join(args.directory, f'{pat}'))) for pat in args.lig_patterns]
        dirs_process = list(set([os.path.dirname(x) for x in dirs_process]))
        dirs_process.sort()

        print(f'dirs_process: {dirs_process}', flush=True)

        for dir in dirs_process:
            ligfiles = []
            [ligfiles.extend(glob.glob(os.path.join(dir, f'{pat}'))) for pat in args.lig_patterns]
            ligfiles = [x for x in ligfiles if not any([pat in os.path.basename(x) for pat in LIG_PATTERNS_EXCLUDE])]
            if len(ligfiles) == 0:
                print(f'\nCouldn\'t find any ligand files using patterns "{args.lig_patterns}" in path "{dir}". Skipping...')
            else:
                ligfiles_process.append(ligfiles)
               
    print(f'\nFound {len(ligfiles_process)} set of ligands to process:')
    if args.verbose:
        for ligfiles in ligfiles_process:
            print('\n'+'\n    '.join(ligfiles))

    inp = '' if not args.yes else 'y'
    while inp not in ['y','n']:
        inp = input(f'\nProcess the above {len(ligfiles_process)} ligand files? (y/n):  ').lower()
    if inp == 'n':
        sys.exit('Aborting...')

    pbar = trange(len(ligfiles_process))
    for i in pbar:
        ligfiles = ligfiles_process[i]
        print(' Processing files...', flush=True)
        print('    '+'\n    '.join(ligfiles))
        for ligfile in ligfiles:
            outfile = ligfile.replace('.sdf', '_opt.sdf')
            if not args.overwrite and os.path.exists(outfile):
                print(f'\nOutput already exists. Skipping: {ligfile}')
                continue
                    
            ligs = BulkSDMolParser(ligfile).get_rdmols(sanitize=False, removeHs=True)
            for i in range(len(ligs)):
                try:
                    ligs[i] = get_largest_fragment(ligs[i])
                except:
                    ligs[i] = Chem.MolFromSmiles('C')
                    pass

            ligs = [get_largest_fragment(m) for m in ligs]
            mols_opt, energy_init, energy_final, success_bools, reasons = optimize_mols_multi(ligs, n_tries=10, max_iters=500, max_attempts=20, num_procs=8, mmff=True)

            print('\nWriting optimized mols to:', outfile)
            write_mols_sdf(mols_opt, outfile)

            df = pd.DataFrame()
            df['ffopt_energy_init'] = energy_init
            df['ffopt_energy_final'] = energy_final
            df['ffopt_success'] = success_bools
            df['ffopt_reason'] = reasons
            df.to_csv(outfile.replace('_opt.sdf', '_ffopt.csv'), index=False)


    t1 = time.time() - t0
    print('\n\nTotal time: %dm %ds'%(t1//60, t1%60))
