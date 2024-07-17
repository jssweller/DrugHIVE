import os
import sys
import gzip
import time
import glob
import math
import random
import multiprocessing
from multiprocessing import Pool, current_process
import shutil
import traceback
from functools import partial
from copy import copy
import h5py
from tqdm import tqdm, trange
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from .molecules import BulkMol2Parser, BulkSDMolParser, MolDatasetGenerator, MolFilter, Mol2, get_mol_stats


TMP_ROOT = '/tmp'


def get_molparser(file_ext):
    if 'sdf' in file_ext:
        return BulkSDMolParser()
    elif 'mol2' in file_ext:
        return BulkMol2Parser()

def get_extension(filepath):
    if '.sdf' in os.path.basename(filepath):
        file_extension = 'sdf'
    elif '.mol2' in os.path.basename(filepath):
        file_extension = 'mol2'
    return file_extension

def check_args(args):
    assert args.proc_id_start > 50, f'Invalid args.proc_id_start: {args.proc_id_start}. As a safety, args.proc_id_start must be set to a value greater than 50. Aborting...'

def combine_hdf5_files(input, output=None, prune=True, delete=False, verbose=True, **kwargs):
    if isinstance(input, str):
        assert os.path.isdir(input), f'Invalid directory: {input}'
        h5_files = glob.glob(os.path.join(input, '*.h5')) + glob.glob(os.path.join(input, '**', '*.h5'), recursive=True)
        h5_files = [x for x in h5_files if not any([pat in x for pat in ['allstats', 'combined']])]
    else:
        assert isinstance(input, list), f'Input must either be a list of files or a directory path. Received {input}'
        h5_files = input
    
    assert len(h5_files) > 0, f'No files found in input path: {input}'
    return _combine_hdf5_files(h5_files, output, prune, delete, verbose, **kwargs)


def _combine_hdf5_files(files, output=None, prune=True, delete=False, verbose=True):
    print('# files to combine:', len(files))
    root = os.path.dirname(files[0])
    h5_files = files
    n_mols_total = 0
    
   
    if verbose:
        print('\nCalculating dataset length...', flush=True)
    idxs_all = []
    n_atoms_max = 0
    n_atom_types_max = 0
    for i in trange(len(h5_files)):
        file = h5_files[i]
        with h5py.File(file) as fnew:
            if prune:
                assert not any(['bond' in key for key in fnew.keys()]), 'Pruning not implemented for bond data! Aborting...'
                idxs = (fnew['atom_types'][:,0,0] != -1)
                idxs_all.append(idxs)
                n_mols_total += sum(idxs)

                n_atoms_max = max(n_atoms_max, get_natoms_max(fnew))
                n_atom_types_max = max(n_atom_types_max, get_atom_types_max(fnew))
            else:
                n_mols_total += len(fnew['atom_types'])
    print('# mols total:', n_mols_total)

    if prune:
        print(f'\npruned "n_atoms_max" shape is {n_atoms_max}')
        print(f'pruned "n_atom_types_max" shape is {n_atom_types_max}')
    
    if output is None:
        savename = os.path.join(root,'combined.h5')
    else:
        if not (output.endswith('.h5') or output.endswith('.h5py')):
            output += '.h5'
        savename = output
    print('\nSaving combined files to:', savename, flush=True)

    j = 0
    with h5py.File(savename, 'w') as f0:
        for i in trange(len(h5_files)):
            file = h5_files[i]
            with h5py.File(file) as fnew:
                if prune:
                    idxs = idxs_all[i]
                    n_dat = sum(idxs)
                else:
                    n_dat = len(fnew['atom_types'])

                for key in fnew.keys():
                    if not key in f0.keys():
                        shape = list(fnew[key].shape)
                        shape[0] = n_mols_total
                        if prune:
                            if key == 'atom_types':
                                shape[1] = n_atoms_max
                                shape[2] = n_atom_types_max
                            if key == 'atom_coords':
                                shape[1] = n_atoms_max
                        print(f'\nCreating dataset "{key}" with shape {shape}', flush=True)
                        
                        chunks = min(1000, n_mols_total)
                        f0.create_dataset(key, 
                                             shape=shape,
                                             chunks=(chunks, *shape[1:]),
                                             compression=fnew[key].compression,
                                             dtype=fnew[key].dtype,
                                             fillvalue=fnew[key].fillvalue)
                    if prune:
                        newdat = fnew[key][idxs]
                        if key == 'atom_types':
                            newdat = newdat[:, :n_atoms_max, :n_atom_types_max]
                        if key == 'atom_coords':
                            newdat = newdat[:, :n_atoms_max]
                    else:
                        newdat = fnew[key][:]

                    f0[key][j:j+n_dat] = newdat
            j += n_dat
            f0.flush()

    if delete:
        print('Removing individual files...')
        for i in trange(len(h5_files)):
            f = h5_files[i]
            if os.path.isfile(f):
                os.remove(f)
            d = os.path.dirname(f)
            if os.path.isdir(d) and (os.path.basename(d) == 'h5') and (len(os.listdir(d)) == 0):
                shutil.rmtree(d)
            



def get_natoms_max(h5_dat, chunks=10000):
    types = h5_dat['atom_types']
    n_atoms_max = 0
    i = 0
    while i < len(h5_dat['atom_types']):
        types = h5_dat['atom_types'][i:i+chunks]
        tmax = np.max(types, axis=0)
    
        val_new = np.arange(tmax.shape[0])[(tmax == -1).all(axis=1)][0]
        n_atoms_max = max(n_atoms_max, val_new)
        assert (types[:, n_atoms_max:, :] != -1).sum()==0, 'Invalid max value returned!'
        i += chunks
    return n_atoms_max


def get_atom_types_max(h5_dat, chunks=10000):
    types = h5_dat['atom_types']
    n_types_max = 0
    i = 0
    while i < len(h5_dat['atom_types']):
        types = h5_dat['atom_types'][i:i+chunks]
        tmax = np.max(types, axis=0)
    
        val_new = np.arange(tmax.shape[1])[(tmax == -1).all(axis=0)][0]
        n_types_max = max(n_types_max, val_new)
        assert (types[:, :, n_types_max:] != -1).sum() == 0, 'Invalid max value returned!'
        i += chunks
    return n_types_max


class MolfileProcessor(object):
    def __init__(self, args, savedir, procnum=0) -> None:
        self.procname = 'processor'
        self.args = args
        self.savedir = savedir
        self.procnum = procnum

        
        i = procnum
        self.tmp_path = f'{TMP_ROOT}/zinctemp{i}.sdf.gz'
        with open(self.tmp_path, 'w+') as f:
            f.write('')

        self.n_processed = 0
        self.n_valid = 0
        self.file_process_times = 0

        self.init_stdout_stderr()


    @property
    def stdout(self):
        return f'{TMP_ROOT}/molfiles_{self.procname}{self.procnum}.stdout'

    @property
    def stderr(self):
        return self.stdout
    
        

    def init_molparser(self, filepath):
        self.args.file_extension = get_extension(filepath)
        self.molparser = get_molparser(self.args.file_extension)


    def set_stdout_stderr(self):
        stdout = self.stdout
        stderr = self.stderr
        if stdout is not None:
            sys.stdout = open(stdout, 'a+')
            if stderr is None:
                sys.stderr = sys.stdout
        if stderr is not None:
            if stderr == stdout:
                sys.stderr = sys.stdout
            else:
                sys.stderr = open(stderr, 'a+')
    

    def init_stdout_stderr(self):
        with open(self.stdout, 'w+') as f:
            f.write('')
        with open(self.stderr, 'w+') as f:
            f.write('')


    def flush_stdout_stderr(self, file_num):
        with open(self.stdout, 'a+') as f:
            f.write(self.stdout_stringio.value)


    def print_summary(self):
        self.n_valid = N_VALID
        self.n_processed = N_PROCESSED
        self.process_time_total = PROCESS_TIME_TOTAL

        print(f'\nProcessing complete for process {self.procnum}')
        print(f'  molecules processed: {self.n_processed}')
        print(f'  molecules valid: %d'%self.n_valid)
        print(f'  time elapsed (total): %dm %ds'%(self.process_time_total//60, self.process_time_total%60), flush=True)

    def close(self):
        time.sleep(.1)
        self.print_summary()



class MolfileStats(MolfileProcessor):
    def __init__(self, args, savedir, procnum=0, molfilter=None) -> None:
        super().__init__(args, savedir, procnum)
        self.procname = 'stats'

        self.molfilter = molfilter
        if self.molfilter is None:
            self.molfilter = MolFilter.init_default()

    def process_file(self, file, file_num):
        self.get_stats(file, file_num)

    def get_stats(self, file, file_num):
        global N_VALID
        global PROCESS_TIME_TOTAL
        global N_PROCESSED

        self.set_stdout_stderr()
        try:
            if not hasattr(self, 'molparser'):
                self.init_molparser(file)

            molparser = self.molparser
            molfilter = self.molfilter
            savedir = self.savedir
            args = self.args
            procnum = self.procnum
            tmp_path = self.tmp_path

            root = os.path.dirname(args.path)
            i_file = file_num


            stats_all = None
            f = file.replace(root,'')
            print(f'\n\nPROCESSING file_num: {file_num} \t FILE: {f}')
            print('path:', file, flush=True)
            
            fname = os.path.basename(file)
            file_path = file

            
            savename = fname.replace('.sdf.gz', '.stats.csv.gz')
            if not args.overwrite:
                if os.path.isfile(os.path.join(savedir,savename)):
                    print('Output file already exists. Skipping...', flush=True)
                    return
                
            shutil.copy(file_path, tmp_path)

            molparser.open(tmp_path)

            n_processed_file = 0
            n_valid_file = 0
            time_file = time.time()
            stats_all = []
            while not molparser.endfile:
                self.n_processed += 1
                N_PROCESSED += 1
                n_processed_file += 1
                
                try:
                    mtext, molname = molparser.get_next_block()
                except:
                    continue
                try:
                    rdmol = molparser.mol_from_mol_block(mtext)
                except:
                    continue
                
                if args.proportion_keep == 1. or (random.random() <= args.proportion_keep):
                    if molfilter.check_mol(rdmol):
                        self.n_valid += 1
                        N_VALID += 1
                        n_valid_file += 1
                        stats = get_mol_stats(rdmol, include_morgan=False)
                        stats['zinc_id'] = int(molname.replace('ZINC',''))
                        if stats['murcko'].iloc[0] == '':
                            stats['murcko'] = 'xx'+Chem.MolToSmiles(rdmol)

                        stats_all.append(stats)

            if len(stats_all) > 0:
                stats_all = pd.concat(stats_all)
                
                if 'morgan_vec' in stats_all.columns:
                    morgan_vals = np.stack(stats_all['morgan_vec'].values)
                    s = savename.replace('.csv.gz','')
                    np.save(os.path.join(savedir, f'{s}.morgan'), morgan_vals)
                    del stats_all['morgan_vec']

                int_keys = [x for x in stats_all.columns if 'int' in str(stats_all[x].dtype) and 'zinc_id' not in x]
                for key in int_keys:
                    stats_all[key] = stats_all[key].astype('int16')

                float_keys = [x for x in stats_all.columns if 'float' in str(stats_all[x].dtype)]
                for key in float_keys:
                    stats_all[key] = stats_all[key].astype('float32')

                
                print(f'Saving `stats_all` to: "%s"'%os.path.join(savedir, savename), flush=True)
                stats_all.to_csv(os.path.join(savedir, savename), index=False, compression='gzip')
            
            print('\nmolecules processed (file): ',n_processed_file)
            print('molecules valid (file): ',n_valid_file)
            t1 = (time.time() - time_file)
            print('time elapsed (file): %dm %ds'%(t1//60, t1%60), flush=True)
            self.file_process_times += t1
            PROCESS_TIME_TOTAL += t1
            # print('time elapsed (total): %dm %ds'%(t1//60, t1%60), flush=True)
            # self.flush_stdout_stderr(file_num)

        except Exception as e:
            traceback.print_exc()
            # self.flush_stdout_stderr(file_num)
            raise e




######### MolfileSampler ###############

def load_stats_hist_data(path):
    hist = np.load(glob.glob(path+'/hist.npy')[0])
    hist_keys = np.load(glob.glob(path+'/hist_keys.npy')[0])

    hist_edges = []
    for key in hist_keys:
        hist_edges.append(np.load(glob.glob(path+f'/{key}.edges.npy')[0]))
    return hist, hist_edges, hist_keys


def filter_hist(hist, hist_edges, hist_keys, molfilter):
    filter_keys = {'mol_width': 'width',
               'num_atoms': 'natoms',
               'hba' : 'hba',
               'hbd' : 'hbd',
               'rotb' : 'rotb',
               'alogp' : 'logp',
              }
    
    new_edges = []
    for i, key in enumerate(hist_keys):
        mfkey = filter_keys[key]
        edges = hist_edges[i]
        dx = edges[1] - edges[0]
        
        idxs = np.arange(len(edges))
        
        min_val = molfilter.filter_params[mfkey+'_min']
        if min_val is not None:
            filter_mask = (edges <= min_val)
            idx_min = idxs[filter_mask]
            if len(idx_min) == 0:
                idx_min = 0
            else:
                idx_min = idx_min[-1]
            
        
        max_val = molfilter.filter_params[mfkey+'_max']
        if max_val is not None:
            filter_mask = (edges > max_val)
            idx_max = idxs[filter_mask]
            if len(idx_max) == 0:
                idx_max = len(edges)
            else:
                idx_max = idx_max[0]
        
        idxs_mask = np.arange(idx_min, idx_max-1)
        hist = np.take(hist, indices=idxs_mask, axis=i)
        new_edges.append(edges[idx_min:idx_max])
        assert hist.shape[i]+1 == len(new_edges[-1]), 'Incorrect shape of histogram edges'
        
    return hist, new_edges


def get_hist_probs(hist):
    hist_probs = copy(hist).astype(float)
    hist_probs[hist > 0] = 1/hist[hist>0]
    assert hist_probs.max() <=1, 'Invalid probability!'
    return hist_probs



def get_n_sample_per_bin(hist, n_sample, n_total=None):
    if n_total is None:
        n_total = int(505 *1e6)  # zinc20 drug_like and within property filter ranges

    proportion_sample = n_sample/n_total
    print('\nproportion_sample:', proportion_sample)

    thresh = 0
    w_prob = .1

    nbins_tot = (hist>0).sum()

    n_sample_eff = hist.sum() * proportion_sample
    print('n_sample_eff:', n_sample_eff)

    while thresh <= w_prob and thresh < 500:
        n_sample2 = copy(n_sample_eff)
        thresh += 1
        nbins = (hist >= thresh).sum()
        if thresh > 1:
    #         n_sample2 = n_sample - hist[(hist>0)*(hist<thresh)].sum()
            n_sample2 = n_sample_eff - (nbins_tot - nbins) * hist[(hist>0)*(hist<thresh)].mean()
        n_per_bin = n_sample2/nbins

    print('\nthresh:', thresh)
    print('n_per_bin', n_per_bin)
    print('\n# nonzero bins:', nbins_tot)
    print('# nonzero bins (above thresh):', nbins)
    return n_per_bin


def get_sample_probs(stats_df, hist_probs, hist_edges, hist_keys):
    bin_idxs = np.zeros((len(stats_df), hist_probs.ndim), dtype=int)
    
    for i, key in enumerate(hist_keys):
        bin_idxs[:, i] = np.searchsorted(hist_edges[i], stats_df[key].values, side='left') - 1

    valid_mask = ~(bin_idxs == np.array(hist_probs.shape)).any(axis=1)  # sample within histogram limits

    bx = bin_idxs[valid_mask]

    probs = hist_probs[tuple(bx[:,i] for i in range(bx.shape[1]))] # probabilities for each valid entry
    return probs


class MolfileSampler(MolfileProcessor):
    def __init__(self, args, savedir, num_sample=None, num_dataset=None, procnum=0, molfilter=None, hist_path=None) -> None:
        super().__init__(args, savedir, procnum)
        self.procname = 'sample'

        self.hist_path = hist_path
        self.n_sampled = 0

        if molfilter is None:
            self.molfilter = MolFilter().init_default()
            self.molfilter.atomic_nums = None
        else:
            self.molfilter = molfilter
        
        # print('\nmolfilter:')
        # print(self.molfilter.filter_params)

        if self.hist_path is not None:
            self.hist, self.hist_edges, self.hist_keys = load_stats_hist_data(hist_path)
            self.hist, self.hist_edges = filter_hist(self.hist, self.hist_edges, self.hist_keys, self.molfilter)
            self.hist_probs = get_hist_probs(self.hist)
            self.n_sample_per_bin = get_n_sample_per_bin(self.hist, self.num_sample, self.num_dataset)
            self.num_sample = num_sample
            self.num_dataset = num_dataset


    def process_file(self, *args, **kwargs):
        return self.sample_file(*args, **kwargs)
    

    def sample_file(self, file, file_num):
        global N_SAMPLED
        global N_VALID
        global PROCESS_TIME_TOTAL
        global N_PROCESSED

        self.set_stdout_stderr()
        try:
            if not hasattr(self, 'molparser'):
                self.init_molparser(file)

            molparser = self.molparser
            molfilter = self.molfilter
            savedir = self.savedir
            args = self.args
            procnum = self.procnum
            tmp_path = self.tmp_path

            root = os.path.dirname(args.path)

            f = file.replace(root,'')
            print(f'\n\nPROCESSING file_num: {file_num} \t FILE: {f}')
            print('path:', file, flush=True)
            
            fname = os.path.basename(file)
            file_path = file

            
            savename_stats = fname.replace('.sdf.gz', '.stats.csv.gz')
            savename_mols = fname.replace('.sdf','.sdf')
            if not args.overwrite:
                if os.path.isfile(os.path.join(savedir,savename_mols)):
                    print('Output file already exists. Skipping...', flush=True)
                    return
            
            shutil.copy(file_path, tmp_path)

            molparser.open(tmp_path)

            n_processed_file = 0
            n_valid_file = 0
            n_sampled_file = 0
            time_file = time.time()

            rdmol_blocks = []
            stats_all = []
            while not molparser.endfile:
                self.n_processed += 1
                N_PROCESSED += 1
                n_processed_file += 1
                
                try:
                    mtext, molname = molparser.get_next_block()
                except:
                    continue
                try:
                    rdmol = molparser.mol_from_mol_block(mtext)
                except:
                    continue
                

                if args.proportion_keep < 1.:
                    if random.random() > args.proportion_keep:
                        continue
                
                if molfilter.check_mol(rdmol):
                    self.n_valid += 1
                    N_VALID += 1
                    n_valid_file += 1

                    if self.hist_path is not None:
                        stats = get_mol_stats(rdmol, keys=self.hist_keys, include_morgan=False)
                        stats['zinc_id'] = int(molname.replace('ZINC',''))
                        if 'murcko' in stats.keys():
                            if stats['murcko'].iloc[0] == '':
                                stats['murcko'] = 'xx'+Chem.MolToSmiles(rdmol)
                        stats_all.append(stats)

                    rdmol_blocks.append(mtext)

            if len(rdmol_blocks) > 0:
                if args.hist_path is not None:
                    probs = get_sample_probs(stats_all, self.hist_probs, self.hist_edges, self.hist_keys)
                    sample_mask = np.random.random(len(probs)) < (probs * self.n_sample_per_bin)  # accept samples randomly
                    rdmol_blocks_save = [m for m, success in zip(rdmol_blocks, sample_mask) if success]
                    stats_all = stats_all.iloc[sample_mask]
                else:
                    rdmol_blocks_save = rdmol_blocks
                    
                n_sampled_file = len(rdmol_blocks_save)
                print('len(rdmol_blocks_save):', len(rdmol_blocks_save))
                if n_sampled_file > 0:
                    N_SAMPLED += n_sampled_file
                    # rdmols_save = [m for m, success in zip(rdmols, sample_mask) if success]

                    ### save rdmols here

                    print(f'Saving mols to: "%s"'%os.path.join(savedir, savename_mols), flush=True)
                    with gzip.open(os.path.join(savedir, savename_mols),'wb+') as outf:
                        for mblock in rdmol_blocks_save:
                            outf.write(mblock.encode())

                    with open(os.path.join(savedir, 'molcounts.txt'), 'a+') as f:
                        f.write(f'{fname} {n_sampled_file}\n')

                    
                    if len(stats_all) > 0:
                        stats_all = pd.concat(stats_all)
                        ### save stats
                        if 'morgan_vec' in stats_all.columns:
                            morgan_vals = np.stack(stats_all['morgan_vec'].values)
                            s = savename_stats.replace('.csv.gz','')
                            np.save(os.path.join(savedir,'stats', f'{s}.morgan'), morgan_vals)
                            del stats_all['morgan_vec']

                        int_keys = [x for x in stats_all.columns if 'int' in str(stats_all[x].dtype) and 'zinc_id' not in x]
                        for key in int_keys:
                            stats_all[key] = stats_all[key].astype('int16')

                        float_keys = [x for x in stats_all.columns if 'float' in str(stats_all[x].dtype)]
                        for key in float_keys:
                            stats_all[key] = stats_all[key].astype('float32')
                    
                        print(f'Saving `stats_all` to: "%s"'%os.path.join(savedir, 'stats', savename_stats), flush=True)
                        stats_all.to_csv(os.path.join(savedir,'stats', savename_stats), index=False, compression='gzip')

                

            print('\nmolecules processed (file): ',n_processed_file)
            print('molecules valid (file): ',n_valid_file)
            print('molecules sampled (file): ',n_sampled_file)
            t1 = (time.time() - time_file)
            print('time elapsed (file): %dm %ds'%(t1//60, t1%60), flush=True)
            self.file_process_times += t1
            PROCESS_TIME_TOTAL += t1

            return (fname, n_sampled_file)

        except Exception as e:
            traceback.print_exc()
            # self.flush_stdout_stderr(file_num)
            raise e

    def print_summary(self):
        self.n_sampled = N_SAMPLED
        self.n_valid = N_VALID
        self.n_processed = N_PROCESSED
        self.process_time_total = PROCESS_TIME_TOTAL

        
        print(f'\nProcessing complete for process {self.procnum}')
        print(f'  molecules processed: {self.n_processed}')
        print(f'  molecules valid: %d'%self.n_valid)
        print(f'  time elapsed (total): %dm %ds'%(self.process_time_total//60, self.process_time_total%60), flush=True)



#### DatasetGenerator


class MolfileDatasetGenerator(MolfileProcessor):
    def __init__(self, args, savedir, procnum=0, molfilter=None, molparser=None) -> None:
        super().__init__(args, savedir, procnum)
        self.procname = 'datagen'

        self.molfilter = molfilter
        if molfilter is None:
            self.molfilter = MolFilter.init_default()
        

    
    def process_file(self, file, file_num):
        global N_VALID
        global N_SAMPLED
        global PROCESS_TIME_TOTAL
        global N_PROCESSED

        self.set_stdout_stderr()
        try:
            if not hasattr(self, 'molparser'):
                self.init_molparser(file)

                self.dataset_gen = MolDatasetGenerator(extension=self.args.file_extension, molparser=self.molparser, molfilter=self.molfilter)



            molparser = self.molparser
            molfilter = self.molfilter
            dataset_gen = self.dataset_gen
            savedir = self.savedir
            args = self.args
            procnum = self.procnum
            tmp_path = self.tmp_path

            root = os.path.dirname(args.path)

            f = file.replace(root,'')
            print(f'\n\nPROCESSING file_num: {file_num} \t FILE: {f}')
            print('path:', file, flush=True)
            
            fname = os.path.basename(file)
            file_path = file

            savename = fname.replace(dataset_gen.extension,'h5')
            if not args.overwrite:
                if os.path.isfile(os.path.join(savedir, savename)):
                    print('Output file already exists. Skipping...', flush=True)
                    return
            
            shutil.copy(file_path, tmp_path)
            # molparser.open(tmp_path)

            time_file = time.time()
            n_processed_file, n_valid_file = dataset_gen.save_coords_types(tmp_path,
                                                                            savename=fname,
                                                                            savedir=savedir, 
                                                                            numstart=0,
                                                                            chunks=1000,
                                                                            numread='all',
                                                                            proportion_read=args.proportion_keep,
                                                                            mol_natoms_max=50,
                                                                            center_mols=True,
                                                                            verbose=True
                                                                            )
            
            N_VALID += n_valid_file
            N_PROCESSED += n_processed_file

           
            



            print('\nmolecules processed (file): ',n_processed_file)
            print('molecules valid (file): ',n_valid_file)
            t1 = (time.time() - time_file)
            print('time elapsed (file): %dm %ds'%(t1//60, t1%60), flush=True)
            self.file_process_times += t1
            PROCESS_TIME_TOTAL += t1

            return (fname, n_valid_file)

        except Exception as e:
            traceback.print_exc()
            raise e

    def print_summary(self):
        self.n_valid = N_VALID
        self.n_processed = N_PROCESSED
        self.process_time_total = PROCESS_TIME_TOTAL

        
        print(f'\nProcessing complete for process {self.procnum}')
        print(f'  molecules processed: {self.n_processed}')
        print(f'  molecules valid: %d'%self.n_valid)
        print(f'  time elapsed (total): %dm %ds'%(self.process_time_total//60, self.process_time_total%60), flush=True)




def main_datagen(args):
    check_args(args)
    molfilter = args.molfilter

    path = args.path
    
    MAX_URLS_DL = 100000

    print('PARENT PID:', os.getpid())
    args.num_procs = min(max(multiprocessing.cpu_count()-2, 1), args.num_procs)  # max=(num_cpu-2), min=1
    print('NUM PARALLEL PROCESSES:', args.num_procs)
    print('proc_id_start:', args.proc_id_start)

    # assert os.path.isfile(path), f'please provide path to download urls file. Invalid path: {path}'
    # files = np.loadtxt(path, dtype='str')

    files = glob.glob(os.path.join(path, '*.sdf.gz'))
    assert len(files) > 0, f'No files found in path: {path}'
    file_nums = np.arange(0, len(files))
    
    args.file_extension = get_extension(os.path.basename(files[0]))
    args.file_extension = 'sdf'
    print('\n\nfile extension:', args.file_extension)

    if args.url_range is None:
        args.url_range = (0,len(files))
    args.url_range = (args.url_range[0], min(len(files), args.url_range[1]))

    print('len files:', len(files))
    print('url_range:', args.url_range)

    files = files[args.url_range[0]:args.url_range[1]]
    file_nums = file_nums[args.url_range[0]:args.url_range[1]]
    
    print('files per process:', int(len(files)/args.num_procs))
    
    root = os.path.dirname(path)
    
    if args.savedir is None:
        args.savedir = os.path.join(args.path, 'h5')
    savedir = args.savedir
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'molcounts.txt'), 'w+') as f:
        f.write('')

    ### filter files here  ###

    n_files_process = len(files)
    n_process_loops = int(np.ceil(n_files_process/MAX_URLS_DL))
    url_ranges_loop = np.linspace(0, len(files), n_process_loops+1, dtype=int)

    print('n_process_loops:', n_process_loops)
    print('url_ranges loop:', url_ranges_loop)
    
    file_processors = []
    for p in range(args.num_procs):
        proc_id = p + args.proc_id_start
        print(f'Process {p} has proc_id: {proc_id}')
        file_processors.append(MolfileDatasetGenerator(args=args, 
                                                savedir=savedir,
                                                procnum=proc_id,
                                                molfilter=molfilter,
                                                ))

    t0 = time.time()
    results = []
    try:
        for loop in range(n_process_loops):
            t0_loop = time.time()

            url_range = (url_ranges_loop[loop], url_ranges_loop[loop+1])
            files_batch = files[url_range[0]:url_range[1]]
            file_nums_batch = file_nums[url_range[0]:url_range[1]]

            ### DOWNLOAD FILES
            print(f'\n\nBegin processing files loop {loop}')
            print('url range', url_range, flush=True)

            ### PROCESS FILES
            t0_loop_process = time.time()
            with Pool(processes=args.num_procs, initializer=worker_init) as pool:
                process_pids = [c.pid for c in multiprocessing.active_children()]
                assert len(process_pids) == args.num_procs, f'Not all process ids accounted for, only found {len(process_pids)} of {args.num_procs}.'
                result = pool.map(partial(process_multi, file_processors=file_processors, process_pids=process_pids), zip(files_batch, file_nums_batch))
                pool.map(partial(close_processor_multi, file_processors=file_processors, process_pids=process_pids), np.zeros(args.num_procs))
            
            results.extend(list(result))

            tt = time.time() - t0_loop_process
            print(f'\nProcess time (loop): %dm %ds'%(tt//60, tt%60))

    
        # write molcounts.txt
        with open(os.path.join(savedir, 'molcounts.txt'), 'w+') as f:
            ntot = sum([x[1] for x in results])
            f.write(f'tot {ntot}')
            for (fname, n_saved) in results:
                if n_saved > 0:
                    f.write(f'{fname} {n_saved}\n')

    except Exception as e:
        close_tmp_files(args.proc_id_start, args.num_procs)
        raise e


    close_tmp_files(args.proc_id_start, args.num_procs)

    # print('\n\nProcessing complete.')
    tt = time.time() - t0_loop_process
    print(f'\nProcess time (total): %dm %ds'%(tt//60, tt%60))
    print('\n## EOF ##.')



##### utility functions

def close_tmp_files(pid_start, num_procs):
    for id in range(pid_start, pid_start + num_procs):
        close_tmp_file(id)


def close_tmp_file(id):
    tmp_path = f'{TMP_ROOT}/zinctemp{id}.sdf.gz'
    print(f'closing file: {tmp_path}')
    if os.path.isfile(tmp_path):
        os.remove(tmp_path)
    try:
        if os.path.isfile(tmp_path.replace('.gz','')):
            os.remove(tmp_path.replace('.gz',''))
    except:
        print('failed to close "%s"'%(tmp_path.replace('.gz','')))
        pass


##### multiprocessing wrapper functions

def process_multi(input, file_processors, process_pids):
    file, file_num = input
    procnum = process_pids.index(current_process().pid)
    fp = file_processors[procnum]
    return fp.process_file(file, file_num)


def close_processor_multi(input, file_processors, process_pids):
    time.sleep(0.1)
    procnum = process_pids.index(current_process().pid)
    fd = file_processors[procnum]
    fd.close()


def worker_init(*args):
    init_globals()
    time.sleep(0.1)

def init_globals(*args):
    global N_SAMPLED
    global N_VALID
    global PROCESS_TIME_TOTAL
    global N_PROCESSED
    global HTTP

    N_SAMPLED = 0
    N_VALID = 0
    PROCESS_TIME_TOTAL = 0
    N_PROCESSED = 0
    HTTP = None
    time.sleep(0.5)



def main_stats(args):
    check_args(args)

    MAX_URLS_DL = int(1e6) # maximum number of urls to process at once

    path = args.path
    if args.save_dir is None:
        args.save_dir = args.path
    
    print('PARENT PID:', os.getpid())
    args.num_procs = min(max(multiprocessing.cpu_count()-2, 1), args.num_procs)  # max=(num_cpu-2), min=1
    print('NUM PARALLEL PROCESSES:', args.num_procs)

    # assert os.path.isfile(path), f'please provide path to download urls file. Invalid path: {path}'
    # files = np.loadtxt(path, dtype='str')

    files = glob.glob(os.path.join(path, '*.sdf.gz'))
    assert len(files) > 0, f'No files found in path: {path}'
    file_nums = np.arange(0, len(files))
    
    if args.url_range is None:
        args.url_range = (0,len(files))
    args.url_range = (args.url_range[0], min(len(files), args.url_range[1]))

    print('len files:', len(files))
    print('url_range:', args.url_range)

    files = files[args.url_range[0]:args.url_range[1]]
    file_nums = file_nums[args.url_range[0]:args.url_range[1]]
    

    root = os.path.dirname(path)
    savedir = args.save_dir
    
    savedir = os.path.join(savedir, 'stats')
    os.makedirs(savedir, exist_ok=True)

    n_files_process = len(files)
    n_process_loops = int(np.ceil(n_files_process/MAX_URLS_DL))
    url_ranges_loop = np.linspace(0, len(files), n_process_loops+1, dtype=int)

    print('n_process_loops:', n_process_loops)
    print('url_ranges loop:', url_ranges_loop)
    
    file_processors = []
    for p in range(args.num_procs):
        proc_id = p + args.proc_id_start
        file_processors.append(MolfileStats(args=args, 
                                            savedir=savedir, 
                                            procnum=proc_id, 
                                             ))

    t0 = time.time()
    try:
        for loop in range(n_process_loops):
            t0_loop = time.time()

            url_range = (url_ranges_loop[loop], url_ranges_loop[loop+1])
            files_batch = files[url_range[0]:url_range[1]]
            file_nums_batch = file_nums[url_range[0]:url_range[1]]

            print(f'\n\nBegin processing files loop {loop}')
            print('url range', url_range, flush=True)

            ### PROCESS FILES
            t0_loop_process = time.time()
            with Pool(processes=args.num_procs, initializer=worker_init) as pool:
                process_pids = [c.pid for c in multiprocessing.active_children()]
                assert len(process_pids) == args.num_procs, f'Not all process ids accounted for, only found {len(process_pids)} of {args.num_procs}.'
                pool.map(partial(process_multi, file_processors=file_processors, process_pids=process_pids), zip(files_batch, file_nums_batch))
                pool.map(partial(close_processor_multi, file_processors=file_processors, process_pids=process_pids), np.zeros(args.num_procs))

            tt = time.time() - t0_loop_process
            print(f'\nProcess time (loop): %dm %ds'%(tt//60, tt%60))

    except Exception as e:
        close_tmp_files(args.proc_id_start, args.num_procs)
        raise e


    close_tmp_files(args.proc_id_start, args.num_procs)

    # print('\n\nProcessing complete.')
    tt = time.time() - t0_loop_process
    print(f'\nProcess time (total): %dm %ds'%(tt//60, tt%60))
    print('\n## EOF ##.')


def main_sample(args):
    check_args(args)
    hist_path = args.hist_path
    num_sample = args.num_sample
    num_dataset = args.num_dataset
    molfilter = args.molfilter

    path = args.path
    
    MAX_URLS_DL = 100000


    print('PARENT PID:', os.getpid())
    args.num_procs = min(max(multiprocessing.cpu_count()-2, 1), args.num_procs)  # max=(num_cpu-2), min=1
    print('NUM PARALLEL PROCESSES:', args.num_procs)
    print('proc_id_start:', args.proc_id_start)

    # assert os.path.isfile(path), f'please provide path to download urls file. Invalid path: {path}'
    # files = np.loadtxt(path, dtype='str')

    files = glob.glob(os.path.join(path, '*.sdf.gz'))
    assert len(files) > 0, f'No files found in path: {path}'
    file_nums = np.arange(0, len(files))
    
    if args.url_range is None:
        args.url_range = (0,len(files))
    args.url_range = (args.url_range[0], min(len(files), args.url_range[1]))

    print('len files:', len(files))
    print('url_range:', args.url_range)

    files = files[args.url_range[0]:args.url_range[1]]
    file_nums = file_nums[args.url_range[0]:args.url_range[1]]
    
    print('files per process:', int(len(files)/args.num_procs))
    
    
    root = os.path.dirname(path)
    
    if args.savedir is None:
        args.savedir = os.path.join(args.path, 'sampled')
    savedir = args.savedir
    
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, 'stats'), exist_ok=True)

    with open(os.path.join(savedir, 'molcounts.txt'), 'w+') as f:
        f.write('')

    if not args.no_hist and args.hist_path is not None:
        hist_path = args.hist_path
        assert os.path.isdir(hist_path), f'Invalid directory path for input `hist_path`: {args.hist_path}'
    
    ### filter files here  ###

    n_files_process = len(files)
    n_process_loops = int(np.ceil(n_files_process/MAX_URLS_DL))
    url_ranges_loop = np.linspace(0, len(files), n_process_loops+1, dtype=int)

    print('n_process_loops:', n_process_loops)
    print('url_ranges loop:', url_ranges_loop)
    
    file_processors = []
    for p in range(args.num_procs):
        proc_id = p + args.proc_id_start
        print(f'Process {p} has proc_id: {proc_id}')
        file_processors.append(MolfileSampler(args=args, 
                                            savedir=savedir, 
                                            procnum=proc_id,
                                            hist_path=hist_path,
                                            num_sample=num_sample,
                                            num_dataset=num_dataset,
                                            molfilter=molfilter,
                                             ))

    t0 = time.time()
    results = []
    try:
        for loop in range(n_process_loops):
            t0_loop = time.time()

            url_range = (url_ranges_loop[loop], url_ranges_loop[loop+1])
            files_batch = files[url_range[0]:url_range[1]]
            file_nums_batch = file_nums[url_range[0]:url_range[1]]

            ### DOWNLOAD FILES
            print(f'\n\nBegin processing files loop {loop}')
            print('url range', url_range, flush=True)

            ### PROCESS FILES
            t0_loop_process = time.time()
            with Pool(processes=args.num_procs, initializer=worker_init) as pool:
                process_pids = [c.pid for c in multiprocessing.active_children()]
                assert len(process_pids) == args.num_procs, f'Not all process ids accounted for, only found {len(process_pids)} of {args.num_procs}.'
                result = pool.map(partial(process_multi, file_processors=file_processors, process_pids=process_pids), zip(files_batch, file_nums_batch))
                pool.map(partial(close_processor_multi, file_processors=file_processors, process_pids=process_pids), np.zeros(args.num_procs))
            
            results.extend(list(result))

            tt = time.time() - t0_loop_process
            print(f'\nProcess time (loop): %dm %ds'%(tt//60, tt%60))

    
        # write molcounts.txt
        with open(os.path.join(savedir, 'molcounts.txt'), 'w+') as f:
            ntot = sum([x[1] for x in results])
            f.write(f'tot {ntot}')
            for (fname, n_saved) in results:
                if n_saved > 0:
                    f.write(f'{fname} {n_saved}\n')
    except Exception as e:
        close_tmp_files(args.proc_id_start, args.num_procs)
        raise e


    close_tmp_files(args.proc_id_start, args.num_procs)

    # print('\n\nProcessing complete.')
    tt = time.time() - t0_loop_process
    print(f'\nProcess time (total): %dm %ds'%(tt//60, tt%60))
    print('\n## EOF ##.')
