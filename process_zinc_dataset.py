import numpy as np
import pandas as pd
import os
import sys
import time
import glob

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from drughive.molecules import BulkMol2Parser, BulkSDMolParser, MolDatasetGenerator, MolFilter, Mol2
from drughive.molfiles import combine_hdf5_files


def get_mol_files(path, mol_patterns):
   
    files = []
    for pattern in mol_patterns:
        files += glob.glob(os.path.join(path, pattern))
    return files


def generate_molcounts(path, molparser, mol_patterns):
    print('\nGenerating molcounts.txt ....\n')
    rdfiles = get_mol_files(path, mol_patterns)
    print(rdfiles)

    molcounts = []
    for i, file in enumerate(rdfiles):
        f = file.replace(path,'')
        nmols = molparser.count_mols(file)
        molcounts.append([f, nmols])
        print('%d of %d \tcount: %d'%(i+1,len(rdfiles),nmols))
        
    np.savetxt(os.path.join(path,'molcounts.txt'), molcounts, fmt='%s')


def load_molcounts(root):
    if not os.path.isfile(os.path.join(root,'molcounts.txt')):
        return None
    else:
        return np.loadtxt(os.path.join(root,'molcounts.txt'), dtype=str)
    
def update_completed_file(root, savedir, proportion_keep=1., numread='all', verbose=False):
    rdfiles = [x for x in os.listdir(root) if x.endswith('.mol2.gz')]

    molcounts = load_molcounts(root)
    molfiles = molcounts[:,0] # filenames

    for f in rdfiles:
        if f not in molfiles: 
            print('missing file: %s'%(f))
    
    completed = []
    for (f, nmols) in molcounts:
        nmols = int(nmols)
        if numread != 'all':
            nmols = min(numread, nmols)

        nmols = int(nmols * proportion_keep)
        if verbose:
            print(f, nmols, end='\t')
        file = os.path.join(root,f)

        sfiles = glob.glob(os.path.join(savedir,f+'*'))
        if len(sfiles) == 0:
            if verbose:
                print()
            continue
        snums = [int(x[x.rfind('_')+1:x.rfind('.')]) for x in sfiles]
        maxidx = np.argmax(snums)
        maxnum = snums[maxidx]
        maxfile = sfiles[maxidx]
        scount = maxnum + np.load(os.path.join(savedir,maxfile))['names'].size
        if scount < nmols:
            if verbose:
                print('%d of %d complete'%(scount,nmols))
        else:
            if verbose:
                print('done')
            completed.append(f)
    np.savetxt(os.path.join(root,'completed.txt'), completed, fmt='%s')
    return completed


def get_molparser(file_ext):
    if 'sdf' in file_ext:
        return BulkSDMolParser()
    elif 'mol2' in file_ext:
        return BulkMol2Parser()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to extract. Could be single file or directory.')
    parser.add_argument('-p', '--proportion_keep', required=False, default=1., type=float, help='Proportion of mols to keep from each file')
    parser.add_argument('-ext', '--file_extension', required=False, default='', type=str, help='file extension of mol files to extract data from')
    parser.add_argument('-o', '--output', required=False, default='None', type=str, help='output file path with `.h5` or `.h5py` file extenstion')
    parser.add_argument('--nofilter', action='store_true', type=bool, help='Whether to filter for drug-like molecule properties.')


    args = parser.parse_args()
    path = args.path

    valid_exts = ['sdf', 'mol2']

    if os.path.isfile(path):
        files = [path]
        for ext in valid_exts:
            if ext in os.path.basename(path):
                args.file_extension = ext
    else:
        
        while args.file_extension not in valid_exts:
            print('file extension must be specified from list %s'%str(valid_exts))
            args.file_extension = input('Enter file extension: ')

        mol_patterns = ['*.%s*'%args.file_extension]
        files = get_mol_files(path, mol_patterns=mol_patterns)
        exclude_patterns=['combined']
        files = [f for f in files if not any([p in f for p in exclude_patterns])]

    if os.path.isdir(path):
        root = path
    else:
        root = os.path.dirname(path)

    savedir = os.path.join(root,'h5')

    molparser = get_molparser(args.file_extension)
    if args.nofilter:
        molfilter = MolFilter()
    else:
        molfilter = MolFilter(weight_min=200, weight_max=500, width_max=20, check_3d=True, logp_max=5., atomic_nums=[6,7,8,9,15,16,17,35,53])
    dataset_gen = MolDatasetGenerator(extension=args.file_extension, molparser=molparser, molfilter=molfilter)

    molcounts = load_molcounts(root)
    if molcounts is None:
        generate_molcounts(root, molparser, mol_patterns)
        molcounts = load_molcounts(root)
    
    print('Getting completed file list...')
    completed_files = update_completed_file(root, savedir, verbose=False)
        
    t0 = time.time()
    nprocessed = 0
    for i,file in enumerate(files[:]):
        f = file.replace(root,'')
        print('\n\nPROCESSING %d of %d\t FILE: %s'%(i+1,len(files),f))
        print('path:', file)
        if f in completed_files:
            continue
        
        print('Getting completed file list...')
        completed_files = update_completed_file(root, savedir, proportion_keep=args.proportion_keep, verbose=False)
        n_mols_remaining = int(sum([int(c) for mf,c in molcounts if mf not in completed_files]) * args.proportion_keep)
        
        numstart = 0
        if f in completed_files:
            print('%s already done, skipping...'%file)
            continue
        
        # get starting position for file
        sfiles = glob.glob(os.path.join(savedir,f+'*'))
        if len(sfiles) > 0:
            snums = [int(x[x.rfind('_')+1:x.rfind('.')]) for x in sfiles]
            maxidx = np.argmax(snums)
            numstart = snums[maxidx]
            print('File partially processed. Getting start molecule...', numstart)
            n_mols_remaining -= numstart
            
        
        n_mols_processed, n_mols_valid = dataset_gen.save_coords_types(file, 
                                                                    savedir=savedir, 
                                                                    numstart=numstart, 
                                                                    chunks=1000,
                                                                    numread='all',
                                                                    proportion_read=args.proportion_keep,
                                                                    mol_natoms_max=40,
                                                                    center_mols=True,
                                                                    verbose=True
                                                                    )
        
        nprocessed += n_mols_processed
        t1 = time.time() - t0
        t_rem = t1/nprocessed * (n_mols_remaining - nprocessed)
        print('elapsed time: %dm%ds'%(t1//60,t1%60))
        print('time remaining: %dm%ds'%(t_rem//60, t_rem%60))


    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(root, 'zinc_data.h5')
    combine_hdf5_files(savedir, args.output, prune=True, delete=True, verbose=True)
        