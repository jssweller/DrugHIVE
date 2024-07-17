#!/usr/bin/python3

from openbabel import pybel
import numpy as np
import pandas as pd
import os
from os.path import join, dirname, basename
import sys
import glob
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

from drughive.molecules import BulkSDMolParser, MolFilter, MolParser, get_mol_stats, write_mols_sdf
from drughive.trainutils import Hparams
from drughive.generating import MolGenerator


def get_previous_opt_df(dfgen):
    dfopt = dfgen.loc[dfgen.model.str.contains('_opt')]
    if len(dfopt) > 0:
        dfopt['optnum'] = dfopt.model.apply(lambda x: int(x.split('_opt')[-1]))
        dfprev = dfopt.loc[dfopt.optnum == dfopt.optnum.max()]
    else:
        dfprev = dfgen
        dfprev['optnum'] = 0
    return dfprev


def ClusterFps(fps, cutoff=0.2):
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True, reordering=True)
    return cs


def get_rdmol_clusters_df(df):
    rdmols = df.rdmol.tolist()
    [Chem.GetSSSR(m) for m in rdmols]
    cfps = [AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in rdmols]
    cs = ClusterFps(cfps, cutoff=0.7)
    
    df['cluster'] = -1
    df = df.reset_index()
    for ci, c in enumerate(cs):
        for i in c:
            df.loc[i, 'cluster'] = ci
    return df  


def get_next_gen_df(dfgen, sort_key, nbest=5, affinity_quantile_thresh=0.5, opt_increase=True, cluster=True, docking_cmd='qvina', molfilter=None):
    ascending = not opt_increase
    
    if docking_cmd == 'smina':
        aff_key = 'affinity_smina'
    elif 'qvina' in docking_cmd:
        aff_key = 'affinity_qvina'

    df_thresh = dfgen[dfgen.model.str.contains('initial')].filter(['pdb_id',aff_key]).groupby(['pdb_id']).quantile(affinity_quantile_thresh) # choose from best portion of affinity
    
    dfprev = get_previous_opt_df(dfgen)
    if molfilter:
        dfprev['filter_pass'] = dfprev.rdmol.apply(lambda m: molfilter.check_mol(m))
        dfprev = dfprev.loc[dfprev.filter_pass]

    dfbest = pd.DataFrame()
    if cluster:
        # cluster mols
        dfprev = dfprev.groupby(['model','pdb_id'], as_index=False).apply(get_rdmol_clusters_df)
        dfprev = dfprev.groupby(['model', 'cluster', 'pdb_id'], as_index=False).apply(lambda x: x.sort_values(sort_key).reset_index(drop=True))
        dfprev['cluster_idx'] = [x[1] for x in dfprev.index]
        dfprev = dfprev.reset_index(drop=True)

        # sample clustered mols
        for i, x in dfprev.groupby(['model','pdb_id']):
            counts = x.groupby(['cluster']).count().run_name.values
            if len(counts) < nbest:
                xsamp = x.loc[(x.cluster_idx == 0)]
                dfbest = pd.concat([dfbest, xsamp])
                n_tot = len(xsamp)
                ci = 1
                while n_tot < nbest:
                    ci += 1
                    xsamp = x.loc[(x.cluster_idx == ci)]
                    if len(xsamp) > (nbest - n_tot):
                        xsamp = xsamp.sort_values(sort_key, ascending=ascending).iloc[:(nbest - n_tot)]
                    n_tot += len(xsamp)
                    dfbest = pd.concat([dfbest, xsamp])
                    
            elif len(counts) >= nbest:
                xsamp = x.loc[(x.cluster_idx == 0)].sort_values(sort_key, ascending=ascending).iloc[:nbest]
                dfbest = pd.concat([dfbest, xsamp])
    else:
        for label, dfprev in dfprev.groupby(['pdb_id']):
            dfprev = dfprev.loc[dfprev[aff_key] < df_thresh.loc[label, aff_key]]
            dfbest = pd.concat([dfbest, dfprev.sort_values(sort_key, ascending=ascending).iloc[:nbest]])
    return dfbest


def save_next_gen(optnum, dfbest, df_inp, key_opt, savedir):
    print('optnum:', optnum)

    savemols_dir = join(savedir, 'mols_parent', f'mols_{key_opt}_opt_{optnum-1}')
    os.makedirs(savemols_dir, exist_ok=True)
    print('writing mols to:', savemols_dir)

    inputs_file = join(savedir, 'mols_parent', f'opt_{key_opt}_input{optnum}.txt')
    print('writing inputs file to:', inputs_file)

    with open(inputs_file, 'w+') as f:        
        for label, df in dfbest.groupby(['pdb_id']):
            pdb_id = label

            for i in range(len(df)):
                molfile = join(savemols_dir, f'{pdb_id}_mol_best_{i}.sdf')
                recpath = df_inp.loc[df_inp.pdb_id == pdb_id, 'recpath'].values[0]

                write_mols_sdf([df.rdmol.iloc[i]], file=molfile)
                f.write(f'{recpath} {molfile}\n')
    return inputs_file


def get_stats_df(molfiles, calc_stats=False):
    dfstats = pd.DataFrame()
    pbar = tqdm(molfiles)
    for file in pbar:                        
        path = Path(file)
        
        file_smina = file.replace('.sdf', '_smina.csv')
        file_qvina = file.replace('.sdf', '_qvina.csv')

        if not (os.path.isfile(file_smina) or os.path.isfile(file_qvina)):
            print(f'No docking files found. Skipping: {file}')
            continue
        
        pbar.set_description(path.parts[-2])
        df = pd.DataFrame()
        if os.path.isfile(file_smina):
            try:            
                # load docking metrics
                df = pd.concat([df, pd.read_csv(file_smina, header=0)])
            except Exception as e:
                if isinstance(e, pd.errors.EmptyDataError):
                    warnings.warn(f'Warning: empty .csv file for sdf: {file}')
                else:
                    print(f'Something failed with file: {file}')
                    raise e


        if os.path.isfile(file_qvina):
            try:
                dfq = pd.read_csv(file_qvina, header=0)
                df['affinity_qvina'] = dfq['affinity_qvina']
            except Exception as e:
                if isinstance(e, pd.errors.EmptyDataError):
                    warnings.warn(f'Warning: empty .csv file for sdf: {file}')
                else:
                    print(f'Something failed with file: {file}')
                    raise e

        sdparse = BulkSDMolParser(file)
        rdligs = sdparse.get_rdmols(sanitize=False)

        # generate molstats.csv if it doesn't already exist
        sfile = file.replace('.sdf', '.molstats.csv')
        if 'opt.sdf' in basename(file):
            sfile = file.replace('_opt.sdf', '.molstats.csv')
        if os.path.isfile(sfile) and not calc_stats:
            rdstats = pd.read_csv(sfile, header=0)
        else:
            rdstats = get_mol_stats(rdligs)
            rdstats.to_csv(sfile, index=False)
        df = pd.concat([df, rdstats], axis=1) 
        
        df['ffopt_success'] = False
        if '_opt.sdf' in basename(file):
            ffopt_file = file.replace('_opt.sdf', '_ffopt.csv')
            if os.path.isfile(ffopt_file):
                ffopt_stats = pd.read_csv(ffopt_file, header=0)
                if 'success' in ffopt_stats.columns:
                    ffopt_stats['ffopt_success'] = ffopt_stats['success']
                    del ffopt_stats['success']
                df = df.drop(labels=df.columns.intersection(ffopt_stats.columns), axis=1).merge(ffopt_stats, left_index=True, right_index=True)
                df.loc[:, 'ffopt_success'] = df.ffopt_success.astype('bool')

        df['rdmol'] = rdligs 

        df['fname'] = path.parts[-1]
        df['pdb_id'] = path.parts[-2]
        df['experiment'] = path.parts[-3]
        df['model'] = path.parts[-4]

        dfstats = pd.concat([dfstats, df])

    dfstats = dfstats.reset_index()

    dfstats['run_name'] = dfstats.model + ' '+dfstats.experiment + ' ' + dfstats.fname
    dfstats.loc[dfstats.fname.str.contains('_opt') , 'run_name'] = dfstats.run_name.loc[dfstats.fname.str.contains('_opt')] + ' (opt)'
    dfstats.loc[~dfstats.fname.str.contains('_opt') , 'run_name'] = dfstats.run_name.loc[~dfstats.fname.str.contains('_opt')] + ''
    dfstats.loc[:, 'run_name'] = dfstats.run_name.str.replace('_qvina.csv','')
    dfstats.loc[:, 'run_name'] = dfstats.run_name.str.replace('_res2_v2','')
    dfstats.loc[:, 'run_name'] = dfstats.run_name.str.replace('posterior','pt')
    dfstats.loc[:, 'run_name'] = dfstats.run_name.str.replace('prior','pr')
    dfstats.loc[:, 'run_name'] = dfstats.run_name.str.replace('mols_gen', '')
    dfstats.loc[:, 'run_name'] = dfstats.run_name.apply(lambda x: x.split('pr ')[0] if '(opt)' not in x else x.split('pr ')[0] + ' (opt)')
    dfstats.loc[:, 'run_name'] = dfstats.run_name.apply(lambda x: x.strip().replace('fullatom_joint','faj'))
    return dfstats


def get_gen_ref_df(molfiles):
    dfstats = get_stats_df(molfiles)
    df_ref = dfstats.loc[dfstats.fname.str.contains('_ref')]
    dfgen = dfstats.loc[dfstats.fname.str.contains('_gen') & ~ dfstats.fname.str.contains('_ref')]
    return dfgen, df_ref, dfstats


def dock_dir(d, rec_path=None, rec_dir=None, dock_cmd='qvina2.1', overwrite=True, lig_patterns=None, protonate=False):
    '''Runs virtual docking of ligands in directory.'''
    assert rec_path or rec_dir, 'Must provide either `rec_path` or `rec_dir` as input.'
    docking_script = os.path.abspath('dock.py')

    if rec_path:
        cmd_run = ['python', docking_script, dock_cmd, '-r', rec_path, '-d', d, '--yes']
    else:
        cmd_run = ['python', docking_script, dock_cmd, '-rd', rec_dir, '-d', d, '--yes']

    if overwrite:
        cmd_run += ['--overwrite']
    if protonate:
        cmd_run += ['--protonate']

    if lig_patterns is not None:
        assert isinstance(lig_patterns, list), 'input `lig_patterns` must be a list.'
        cmd_run += ['-lp'] + lig_patterns

    result = subprocess.run(cmd_run, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stderr = result.stderr.decode()
    msg_e = None
    if 'FileNotFoundError' in stderr and dock_cmd in stderr:
        msg_e = f'Docking command not found: "{dock_cmd}"'
    elif 'No .pdbqt file found' in result.stdout.decode():
        msg_e = 'No .pdbqt file found!'
    elif result.returncode != 0:
        msg_e = f'return code {result.returncode}'
    if msg_e:
        print('\n\nERROR: docking failed')
        print('\nstderr:')
        print(result.stderr.decode())
        print('\nstdout:')
        print(result.stdout.decode())
        raise Exception(f'Docking Failed: {msg_e}')
    return result


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
    
    #################### Select next generation subset
    
    n_cycles = gargs.n_cycles
    opt_num_curr = 0

    root = gargs.output
    save_name = gargs.save_name
    savedir = join(root, save_name)
    os.makedirs(savedir, exist_ok=True)
    initial_dir = join(root, f'pdbzinc_initial')

    initial_input_file = join(root, 'input.txt')
    with open(initial_input_file, 'w+') as f:
        f.write(gargs['target_path'] + ' ' + gargs['ligand_path'])  

    key_opt = gargs.key_opt
    cluster_parents = gargs.cluster_parents
    opt_increase = gargs.opt_increase
    dock_cmd = gargs.get('docking_cmd','qvina2.1')
    ffopt_mols = gargs.get('ffopt_mols', True)
    protonate = gargs.get('protonate', False)
    n_best_parents = gargs.n_best_parents
    affinity_quantile_thresh = gargs.affinity_quantile_thresh    

    zbetas = gargs.get('zbetas', None)
    temps = gargs.get('temps', 1.)
    if isinstance(zbetas, (int, float)):
        zbetas = [zbetas for _ in range(n_cycles)]
    elif len(zbetas) != n_cycles:
        zbetas = [zbetas for _ in range(n_cycles)]

    
    if isinstance(temps, (int, float)):
        temps = [temps for _ in range(n_cycles)]
    elif len(temps) != n_cycles:
        temps = [temps for _ in range(n_cycles)]

    
    molgen = MolGenerator(gargs.checkpoint, gargs.model_id, random_rot=gargs.random_rotate, random_trans=gargs.random_translate, ffopt=ffopt_mols)
    molfilter = MolFilter(ring_sizes=gargs.get('ring_sizes', None), 
                          ring_system_max=gargs.get('ring_system_max', None), 
                          ring_loops_max=gargs.get('ring_loops_max', None), 
                          double_bond_pairs=gargs.get('dbl_bond_pairs', None),
                          natoms_min=gargs.get('n_atoms_min', None))
    
    if not os.path.isdir(initial_dir):
        print('Generating initial molecules and saving to:', dirname(initial_dir), flush=True)
        molgen.generate_samples(gargs.n_samples_initial, 
                                temps=gargs.get('temps_initial', 1.),
                                zbetas=gargs.get('zbetas_initial', 1.), 
                                input_data_file=initial_input_file,
                                pdb_id=gargs['pdb_id'],
                                savedir=initial_dir,
                                molfilter=molfilter,
                                ffopt=False
                                )
        

    while opt_num_curr <= n_cycles:
        dirs = [d for d in glob.glob(join(savedir,'*')) if 'mols_parent' not in basename(d)]
        opt_nums = [int(basename(d).split('_opt')[-1]) for d in dirs]
        if len(opt_nums) > 0:
            opt_num_prev = max(opt_nums)
        else:
            opt_num_prev = 0
        opt_num_curr = opt_num_prev + 1

        if not initial_dir in dirs:
            dirs.append(initial_dir)
        print('dirs:\n    '+'\n    '.join(dirs))

        # check all mols optimized and docked.
        for d in dirs:
            # get files initial
            fs = glob.glob(join(d, '**', '*.sdf'), recursive=True)
            fs = [f for f in fs if not any([p in f for p in ['_smina.sdf', '_qvina.sdf']])]
            fs_gen = [f for f in fs if not '_opt.sdf' in f]
            fs_gen_opt = [f for f in fs if '_opt.sdf' in f]

            # check all optimized
            if ffopt_mols:
                fs_unoptimized = [f.replace('.sdf','_opt.sdf') for f in fs_gen if not os.path.isfile(f.replace('.sdf','_opt.sdf'))]
                if len(fs_unoptimized) > 0:
                    print(f'\nFound unoptimized mols in dir. Optimizing dir: {d}')
                    print('    '+'\n    '.join(fs_unoptimized), flush=True)
                    opt_script = os.path.abspath('ff_optimize.py')
                    cmd = f'python {opt_script} -d {d} --yes'
                    subprocess.run(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)

            # check all docked
            if 'qvina' in dock_cmd:
                fs_undocked = [f for f in fs if not os.path.isfile(f.replace('.sdf', '_qvina.csv'))]
            elif 'smina' in dock_cmd:
                fs_undocked = [f for f in fs if not os.path.isfile(f.replace('.sdf', '_smina.csv'))]

            dock_lig_patterns = ['lig_ref.sdf', 'mols_gen.sdf']
            if ffopt_mols:
                if opt_num_curr == 0:
                    dock_lig_patterns = ['lig_ref*.sdf', 'mols_gen_opt.sdf']
                else:
                    dock_lig_patterns = ['lig_ref.sdf', 'mols_gen_opt.sdf']

            if len(fs_undocked) > 0:
                print('Docking directory:', d, flush=True)
                result = dock_dir(d, rec_path=gargs['target_path_pdbqt'], dock_cmd=dock_cmd, overwrite=False, lig_patterns=dock_lig_patterns, protonate=protonate)
                if 'UserWarning: Could not find any receptor file' in result.stderr.decode():
                    msg = result.stderr.decode()
                    msg = msg[msg.find('UserWarning:'):]
                    msg = msg[:msg.find('\n')]
                    raise Exception(f'Docking failed with message: {msg}')
                print('Docking complete.', flush=True)
        
        print('\n Loading files for next iteration.')
        print('dirs:\n    '+'\n    '.join(dirs))
        molfiles = []
        for d in dirs:
            # get files
            fs = glob.glob(join(d, '**', '*.sdf'), recursive=True)
            fs = [f for f in fs if not any([p in f for p in ['_smina.sdf', '_qvina.sdf']])]
            if ffopt_mols:
                fs = [f for f in fs if '_opt.sdf' in f]
            else:
                fs = [f for f in fs  if not '_opt.sdf' in f]
            molfiles.extend(fs)
        
        print('\nmolfiles:\n    '+'\n    '.join(molfiles))
        dfgen, df_ref, dfstats = get_gen_ref_df(molfiles)
        dfgen = dfgen.loc[dfgen.ffopt_success]  # filter for successful force field optimization

        if opt_num_curr > n_cycles:
            break

        dfbest = get_next_gen_df(dfgen, 
                                 key_opt, 
                                 nbest=n_best_parents, 
                                 affinity_quantile_thresh=affinity_quantile_thresh,
                                 opt_increase=opt_increase,
                                 cluster=cluster_parents,
                                 docking_cmd=dock_cmd,
                                 molfilter=molfilter
                                 )

        # load initial input file for receptor path
        df_inp = pd.read_csv(initial_input_file, header=None, names=['recpath','ligpath'], delimiter=' ')
        df_inp['pdb_id'] = gargs['pdb_id']

        inputs_file = save_next_gen(opt_num_curr, dfbest, df_inp, save_name, savedir)

        ################ Generate examples    
        
        n_samples = gargs.n_samples

        molgen.model_id = gargs.model_id + f'_{save_name}_opt{opt_num_curr}'

        print(f'Generating children for generation {opt_num_curr}')
        molgen.generate_samples(n_samples, 
                                temps=temps[opt_num_curr-1], 
                                zbetas=zbetas[opt_num_curr-1], 
                                savedir=join(savedir,molgen.model_id), 
                                input_data_file=inputs_file,
                                pdb_id=gargs['pdb_id'],
                                molfilter=molfilter,
                                ffopt=False
                                )
        
