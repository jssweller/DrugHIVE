#!/usr/bin/python3

import glob, sys, os, gzip
import subprocess
from tqdm import tqdm, trange
import itertools
import tempfile
import pandas as pd
import numpy as np
import time
from os.path import join, basename, dirname
import warnings
from copy import copy

from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from multiprocessing import Pool

global PDB_DIRS

PDB_DIRS = ['data/PDBbind_v2020_refined_all/refined-set', 
            'data/PDBbind_v2020_general_minus_refined/v2020-other-PL']

def not_dollars(line):
    ''' Check to see if the input line is the end of the molecule.'''
    return "$$$$" != line.strip('\n')


def parse_molecules(file):
    '''
    A generator function that reads in sdf files one molecule at a time.
    Can also read in a compressed .sdf.gz file
    '''

    if file.endswith('.sdf'): 
        with open(file) as lines:
            while True:
                block = list(itertools.takewhile(lambda x: not x.strip().startswith('$$$$'), lines))
                if not block:
                    break
                yield block + ['$$$$\n']
                
    elif file.endswith('.sdf.gz'): 
        with gzip.open(file) as lines:
            while True:
                block = list(itertools.takewhile(lambda x: not x.strip().startswith('$$$$'), lines))
                if not block:
                    break
                yield block + ['$$$$\n']    

    elif file.endswith('.pdbqt'): 
        with open(file) as lines:
            while True:
                block = list(itertools.takewhile(lambda x: not x.strip().startswith('ENDMDL'), lines))
                if not block:
                    break
                block = [x for x in block if not (x.strip().startswith('MODEL') or x.strip() == '')]
                yield block + ['\n']


def splitligs(ligfile):
    '''
    Split the input sdf file into the specified
    number of temp files. They are spread evenly
    among the partitions with the first ligand in
    the input going into the first partition and
    the second ligand going into the second partition
    and so on.
    
    Arguments:
    ligfile -- ligand file (.sdf)
    num -- number of file to split into
    
    Returns:
    tmpfiles -- list of temporary files
    '''    

    ligfiles = list(parse_molecules(ligfile))
    ligfiles = [''.join(x) for x in ligfiles if len(x) > 1]
    return ligfiles

def get_suffix(ligfile):
    suffix = None
    if ligfile.endswith('.sdf') or ligfile.endswith('.sdf.gz'):
        suffix = '.sdf'
    elif ligfile.endswith('.pdbqt'):
        suffix = '.pdbqt'
    assert suffix is not None, f'Invalid ligfile: {ligfile}'
    return suffix


def get_tmp_ligfile(lig_text, file_type):
    #initialize the temporary files
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix=file_type, dir='/tmp/',  delete=False)
       
    #for each compound in the sdf file, write to each tmp file
    tmpfile.write(''.join(lig_text))
    tmpfile.write('\n')
    tmpfile.close()    
    return tmpfile


def reassemble_ligs(outfile, procfiles):
    '''
    De-interleave the results from docking.
    Takes the first compound from the first file and so on.
    
    Arguments:
    outfile -- output file name
    minfiles -- docking processed names
    
    Returns:
    outfile -- output file name
    '''
    
    # open the minimized files and get the parse_molecules generator for each
    generators = [parse_molecules(procfile.name) for procfile in procfiles]
    
    if os.path.splitext(outfile)[1] == '.gz':
        fout = gzip.open(outfile, 'w')
    else:
        fout = open(outfile, 'w')
        
    for genny in generators:
        for lig in genny:
            fout.write(''.join(lig))
    fout.close()
    return outfile


def sep_mol_largest_frag(molfile):
    with Chem.rdmolfiles.SDMolSupplier(molfile) as supp:
        mol = next(supp)
    if len(Chem.MolToSmiles(mol).split('.')) > 1:
        mol = list(sorted(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False), key=lambda x: len(x.GetAtoms())))[-1]
        with open(molfile, mode='w+') as f, Chem.rdmolfiles.SDWriter(f) as writer:
            writer.SetKekulize(False)
            writer.write(mol)
    return mol


def get_largest_fragment(mol : Chem.Mol, return_frags : bool = False, verbose : bool = False):
    """Separate and return largest fragment in molecule. Fragment is defined as largest connected substructure. If return_frags is true, returns remaining fragments as a separate Chem.Mol
    """    
    ## Another way
    # mols = Chem.rdmolops.GetMolFrags(mol_pred, asMols=True)

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


def is_pdb_id(text, pdb_dirs=None):
    if pdb_dirs is None:
        pdb_dirs = PDB_DIRS
        
    pdb_dirs = [os.path.expanduser(d) for d in pdb_dirs]
    assert any([os.path.isdir(d) for d in pdb_dirs]), f'pdb_dirs are messed up.\ntext (pdb_id):{text}\npdb_dirs: {pdb_dirs}\ncwd: {os.getcwd()}'

        
    return any([len(glob.glob(join(d, text))) > 0 for d in pdb_dirs])


def get_pdb_dirs(pdb_id, pdb_dirs=None):
    if pdb_dirs is None:
        pdb_dirs = PDB_DIRS
    pdb_dirs = [os.path.expanduser(d) for d in pdb_dirs]

    assert all([os.path.isdir(d) for d in pdb_dirs]), 'pdb_dirs are messed up.'

    outdir = []
    for d in pdb_dirs:
        outdir += glob.glob(join(d, pdb_id))
    return outdir


def get_pdb_from_path(path, pdb_dirs=None):
    path0 = path
    while (not is_pdb_id(basename(path), pdb_dirs)) and (len(basename(path)) > 0):
        path = dirname(path)
    pdb_id = basename(path)
    assert is_pdb_id(pdb_id, pdb_dirs), 'No valid pdb_id found in directory path: %s'%path0
    return pdb_id


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, protonate=False):
    cmd = f'obabel {sdf_file} -O {pdbqt_outfile}'
    if protonate:
        cmd += ' -p 7.4'
    subprocess.run(cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return pdbqt_outfile


def pdbqt_to_sdf(pdbqt_file, sdf_outfile):
    subprocess.run(f'obabel {pdbqt_file} -O {sdf_outfile}'.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return sdf_outfile


def qvina_score_only(rec_file, lig_file, data_dict=None, qvina_cmd='qvina2.1', verbose=False):
    cmd = [qvina_cmd, 
        '--receptor', rec_file, 
        '--ligand', lig_file, 
        '--score_only'
        ]
        
    if verbose:
        print('Running score_only cmd:', cmd)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    val_names = {'Affinity:': 'affinity_qvina',
                    'Initial Affinity:': 'affinity_qvina_initial',
                    'Intramolecular energy:': 'intra_e',
                    'gauss 1': 'gauss_1',
                    'gauss 2': 'gauss_2',
                    'repulsion': 'repulsion',
                    'hydrophobic': 'hydrophobic',
                    'Hydrogen': 'hbond',
                    }
    
    if data_dict is None:
        data_dict = defaultdict(list)
    
    vals = {k: float('nan') for k in val_names}

    if len(result.stderr) > 0:
        print('\nstderr qvina_score_only')
        for line in result.stderr.split(b'\n'):
            print(line.decode())

    for line in result.stdout.split(b'\n'):
        line = line.decode().strip()
        for k,v in val_names.items():
            if line.startswith(k):
                try:
                    dat = line.split(':')[1].strip()
                    if '(' in dat:
                        dat = dat.split('(')[0].strip()
                    vals[k] = float(dat)
                except Exception as e:
                    print('ERROR: pattern %s failed\n'%v, '    line:', line, '\n   dat:', dat)

    for k,v in val_names.items():
        data_dict[v].append(vals[k])
    return data_dict

    

def qvina_process_mol(ligfile, recf_pdbqt_name, qvina_cmd='qvina2.1', num_procs=None, score=False, score_only=False, protonate=False, exhaustiveness=None, verbose=False):
    reason = ''
    success = True

    default_vals = pd.DataFrame().from_dict({
        'affinity_qvina': [float('nan')],
        'affinity_qvina_initial': [float('nan')],
        'intra_e': [float('nan')],
        'gauss_1': [float('nan')],
        'gauss_2': [float('nan')],
        'repulsion': [float('nan')],
        'hydrophobic': [float('nan')],
        'hbond': [float('nan')],
        'dock_reason': [reason],
        'dock_success': [success],
        })
    
    if not (score or score_only):
        default_vals = default_vals.filter(['affinity_qvina', 'dock_reason', 'dock_success'])
    
    #initalize another temp file for pdbqt file
    ligf_pdbqt_out = tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', dir='/tmp/',  delete=True)
    ligf_sdf_out = tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', dir='/tmp/',  delete=True)

    if ligfile.name.endswith('.sdf'):
        rdlig = next(Chem.SDMolSupplier(ligfile.name, sanitize=False, removeHs=True))
        if rdlig is None or rdlig.GetNumHeavyAtoms() < 2:
            default_vals['dock_success'] = False
            default_vals['dock_reason'] = 'MOL_NONE'
            return default_vals, None, None
        
        Chem.GetSSSR(rdlig)
        try:
            rdlig = get_largest_fragment(rdlig)
        except Exception as e:
            default_vals['dock_success'] = False
            default_vals['dock_reason'] = 'GET_FRAG'
            return default_vals, None, None
        try:
            ligf = tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', dir='/tmp/',  delete=True)
            with Chem.SDWriter(ligf.name) as writer:
                writer.SetKekulize(False)
                writer.write(rdlig)

            lig_center = rdlig.GetConformer().GetPositions().mean(axis=0)
        except Exception as e:
            print('Writing mol Failed! Skipping....')
            default_vals['dock_success'] = False
            default_vals['dock_reason'] = 'SDWRITER_FAIL'
            return default_vals, None, None
        
        ligf_pdbqt = tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', dir='/tmp/',  delete=True)
        sdf_to_pdbqt(ligf.name, ligf_pdbqt.name, protonate)

    elif ligfile.name.endswith('.pdbqt'):
        if num_atoms_pdbqt(ligfile.name) < 2:
            default_vals['dock_success'] = False
            default_vals['dock_reason'] = 'MOL_NONE'
            return default_vals, None, None
        ligf = ligfile
        ligf_pdbqt = ligf


    if exhaustiveness is None:
        exhaustiveness = 8
    grid_size = 20
    data_dict = defaultdict(list)

    out_sdf = None
    out_pdbqt = None
    if not score_only:
        cmd = [qvina_cmd, 
                '--receptor', recf_pdbqt_name, 
                '--ligand', ligf_pdbqt.name, 
                '--center_x', '%.2f'%lig_center[0],  
                '--center_y', '%.2f'%lig_center[1], 
                '--center_z', '%.2f'%lig_center[2],
                '--size_x', '%.2f'%grid_size,
                '--size_y', '%.2f'%grid_size,
                '--size_z', '%.2f'%grid_size,
                '--exhaustiveness', str(int(exhaustiveness)),
                '--num_modes', str(1),
                # '--local_only',
                # '--cpu', '5',
                '--out', ligf_pdbqt_out.name,
                '--log', '/tmp/test.log',
                ]

        if num_procs is not None:
            cmd += ['--cpu', f'{num_procs}']

        if verbose:
            print('Running cmd:', cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        pdbqt_to_sdf(ligf_pdbqt_out.name, ligf_sdf_out.name)
        with open(ligf_sdf_out.name, 'r') as f_lig:
            out_sdf = f_lig.read().split('$$$$\n')[0]+'$$$$\n'

        if verbose:
            print('qvina done:', result)

        key = None
        line_prev = ''
        aff = float('nan')
        for line in result.stdout.split(b'\n'):
            line = line.decode()
            if '-----+------------+----------+----------' in line_prev and 'Writing output' not in line:
                dat = line.strip().split()
                try:
                    aff = float(dat[1])
                except Exception as e:
                    print('ERROR\n  ', 'line:', line, '\n   dat:', dat)
                    reason = 'DOCK_FAIL'
                    success = False
                    verbose = True
                    aff = float('nan')
                # data_dict['rmsd_qvina'] = float(line[1])
            line_prev = line
        
        if not score:
            data_dict['affinity_qvina'].append(aff)
        elif verbose:
            print('affinity:', aff)
        data_dict['dock_success'].append(success)
        data_dict['dock_reason'].append(reason)


        # check for multi-model .pdbqt file
        with open(ligf_pdbqt_out.name, 'r') as f:
            lines = f.read().split('ENDMDL')[0].split('\n') # keep only first model
        
        # remove "MODEL #" tags from .pdbqt file
        lines = [line for line in lines if not any(line.strip().startswith(x) for x in ['MODEL', 'ENDMDL'])]
        out_pdbqt = '\n'.join(lines)
        with open(ligf_pdbqt_out.name, 'w+') as f:
            f.write(out_pdbqt)


    if score or score_only:
        # record score_only output values.
        ligf = ligf_pdbqt.name if score_only else ligf_pdbqt_out.name

        try:
            data_dict = qvina_score_only(recf_pdbqt_name, ligf, data_dict, qvina_cmd, verbose)
            if verbose:
                print('affinity_score:', data_dict['affinity_qvina'][-1])
        except Exception as e:
            reason = 'DOCK_SCORE_ONLY_FAIL'
            success = False
            verbose = True
        
        if score_only:
            data_dict['dock_success'].append(success)
            data_dict['dock_reason'].append(reason)

    if verbose and key is not None:
        print(f"{'mol #'+str(i):10}", end=' ')
        for k,v in data_dict.items():
            print(f'{k}: {v[-1]:20}', end=' ')

    try:
        df = pd.DataFrame().from_dict(data_dict)
    except Exception as e:
        print(data_dict)
        raise e

    ligfile.close()
    return df, out_sdf, out_pdbqt


def num_atoms_pdbqt(file):
    with open(file) as f:
        n = sum([x.strip().startswith('ATOM') for x in f])
    return n

def qvina_process_multi(ligfile, recfile, outfile, qvina_cmd='qvina2.1', num_procs=5, score=False, score_only=False, protonate=False, verbose=False, pbar=False, exhaustiveness=None, overwrite=True):
    SDF_DUMMY = '\n     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$$$$\n'
    PDBQT_DUMMY = 'ROOT\nATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C\nENDROOT\nTORSDOF 0\n'    

    recf_pdb = recfile
    if recf_pdb.endswith('.pdbqt'):
        recf_pdbqt = recf_pdb
    else:
        for pat in ['*pocket*.pdbqt', '*protein*.pdbqt']:
            recf_pdbqt = glob.glob(join(dirname(recf_pdb), pat))
            if len(recf_pdbqt) > 0:
                recf_pdbqt = recf_pdbqt[0]
                break
            else:
                recf_pdbqt = None

    if recf_pdbqt is None or not os.path.isfile(recf_pdbqt):
        print('No .pdbqt file found:', recf_pdb)
        return

    lig_suffix = os.path.splitext(ligfile.replace('.gz',''))[-1]

    if outfile is None:
        outfile = ligfile.replace(lig_suffix,'.csv')
        if not outfile.endswith('_qvina.csv'):
            outfile = outfile.replace('.csv','_qvina.csv')

    if not overwrite and os.path.exists(outfile):
        msg = f'Output already exists. Skipping: {ligfile}'
        print(msg, flush=True)
        return

    ligfile_data = splitligs(ligfile)
    suffix = get_suffix(ligfile)
    
    
    if verbose:
        msg = f'recfile: {recfile}\nligfile: {ligfile}\nNumber of ligands to score: {len(ligfile_data)}'
        print(msg, flush=True)


    results = []
    out_sdfs = []
    out_pdbqts = []

    if verbose:
        print('\nDocking files...', flush=True)

    if pbar:
        p = trange(len(ligfile_data))
    else:
        p = range(len(ligfile_data))
    for i in p:
        ligtmp = get_tmp_ligfile(ligfile_data[i], suffix)
        result, out_sdf, out_pdbqt = qvina_process_mol(ligtmp, recf_pdbqt, qvina_cmd=qvina_cmd, num_procs=num_procs, score=score, score_only=score_only, protonate=protonate, exhaustiveness=exhaustiveness, verbose=verbose)
        results.append(result)
        out_sdfs.append(out_sdf)
        out_pdbqts.append(out_pdbqt)

    
    if not score_only:
        with open(outfile.replace('.csv','.sdf'), 'w+') as f:
            for x in out_sdfs:
                if x is None:
                    x = SDF_DUMMY
                f.write(x)
        with open(outfile.replace('.csv','.pdbqt'), 'w+') as f:
            for i, x in enumerate(out_pdbqts, 1):
                if x is None:
                    x = PDBQT_DUMMY
                f.write('MODEL %d\n%s%s'%(i,x,'ENDMDL\n\n'))


    df = pd.DataFrame()
    for df2 in results:
        df = pd.concat([df,df2])

    df = df.reset_index(drop=True)
    # if os.path.isfile(statsfile):
    #     df = pd.read_csv(statsfile, header=0)
    
    statsfile = outfile
    try:
        df.to_csv(statsfile, index=False)
    except PermissionError:
        fi = 0
        while os.path.isfile(statsfile.replace('.csv', str(fi)+'.csv')):
            fi+=1
        print('Warning: Standard file was open/locked. Writing to:', statsfile.replace('.csv', str(fi)+'.csv'))
        df.to_csv(statsfile.replace('.csv', str(fi)+'.csv'), index=False)

    if verbose:
        print('\nGathering outputs...', flush=True)

    if verbose:
        print('Processing done.')




if __name__ == '__main__':
    global QVINA_PATH

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dock_path', type=str, help='Path to docking command.')
    parser.add_argument('-d', '--directory', required=False, default=None,  help = 'directory to process containing')
    parser.add_argument('-e', '--exhaustiveness', required=False, type=int, default=None,  help = 'Exhaustiveness parameter in Vina.')
    parser.add_argument('-lp', '--lig_patterns', required = False, nargs='+', default=None,  help = 'list of regex patterns for finding ligand files (separated by spaces).')
    parser.add_argument('-rd', '--rec_dirs', required=False, nargs='+', default=None,  help = 'list of directories containing receptor folders (separated by spaces).')
    parser.add_argument('-r', '--receptor', dest = 'recfile', required=False, default=None, help = 'input receptor file (.pdbqt)')
    parser.add_argument('-l', '--ligand', dest = 'ligfile', required=False, default=None, help = 'input ligand file (.sdf or .sdf.gz)')
    parser.add_argument('-o', '--out', dest = 'outfile', required=False, default=None, help = 'output ligand file (.sdf or .sdf.gz)')
    parser.add_argument('--score_only', required=False, action='store_true', help = 'Only score the input ligands.')
    parser.add_argument('--score', required=False, action='store_true', help = 'Score the input ligands after docking.')
    parser.add_argument('-p', '--protonate', required=False, action='store_true', help = 'Protonate each molecule using openbabel before docking.')
    parser.add_argument('--pbar', required=False, action='store_true', help = 'Display progress bar for each docking file.')
    parser.add_argument('-f', '--overwrite', required=False, action='store_true', help = 'Re-process files if output already exists.')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', help = 'print docking output for each molecule')
    parser.add_argument('-y', '--yes', required=False, action='store_true', help = 'Automatically accept prompts.')
    # parser.add_argument('-c', '--cpu', dest = 'cpus', required=False, type = int, default = mp.cpu_count(), help = 'Number of CPUs to use. Default: detect number in system')

    args = parser.parse_args()

    global LIG_PATTERNS_EXCLUDE
    LIG_PATTERNS_EXCLUDE = ['qvina']       
    
    dock_type = ''
    if 'qvina' in basename(args.dock_path):
        QVINA_PATH = args.dock_path
        dock_type = 'qvina'

    t0 = time.time()

    print('\n\nrec_dirs:', args.rec_dirs)
    print('recfile:', args.recfile)
    if args.recfile is None and args.rec_dirs is None:
        args.rec_dirs = PDB_DIRS
    
    if args.rec_dirs is not None:
        args.rec_dirs = [os.path.expanduser(x) for x in args.rec_dirs]


    if args.directory is None:
        assert args.ligfile is not None, 'Either provide `--directory` or `--ligand`.'
        
        if args.recfile is None:
            pdb_id = get_pdb_from_path(args.ligfile, args.rec_dirs)

            # find receptor file based on pdb_id
            recfile = None
            for rec_dir in args.rec_dirs:
                if pdb_id in os.listdir(rec_dir):
                    recfile = glob.glob(join(rec_dir, pdb_id, f'{pdb_id}*_pocket.pdb'))
                    recfile = recfile[0] if len(recfile) > 0 else None
                    if recfile is None:
                        recfile = glob.glob(join(rec_dir, pdb_id, f'{pdb_id}*_protein.pdb'))
                        recfile = recfile[0] if len(recfile) > 0 else None
                    if recfile is not None:
                        if args.verbose:
                            print(f'Found receptor_file: {recfile}', flush=True)
                        break
            args.recfile = recfile


        if args.recfile is None:
            warnings.warn(f'Could not find any receptor file using pattern "**/{pdb_id}*_pocket.pdb" in paths "{args.rec_dirs}". Skipping...')
        else:
            qvina_process_multi(args.ligfile, args.recfile, 
                                        qvina_cmd=QVINA_PATH, 
                                        outfile=args.outfile, 
                                        num_procs=8, 
                                        score=args.score, 
                                        score_only=args.score_only, 
                                        protonate=args.protonate,  
                                        exhaustiveness=args.exhaustiveness, 
                                        verbose=args.verbose, 
                                        pbar=args.pbar, 
                                        overwrite=args.overwrite)
    else:
        assert args.ligfile is None, 'Conflicting flags. Do not provide --ligand when --directory is provided.'
        if args.lig_patterns is None:
            args.lig_patterns = ['lig_ref*.sdf', 'mols_pred*.sdf', 'mols_gen*.sdf']
        else:
            # remove patterns that would exclude the user provided args.lig_patterns
            LIG_PATTERNS_EXCLUDE = [x for x in LIG_PATTERNS_EXCLUDE if not any([x in p for p in args.lig_patterns])]

        # get directories to process by searching for lig_patterns in directory tree
        dirs_process = []
        [dirs_process.extend(glob.glob(os.path.join(args.directory, '**', pat), recursive=True)) for pat in args.lig_patterns]
        [dirs_process.extend(glob.glob(os.path.join(args.directory, pat))) for pat in args.lig_patterns]
        dirs_process = list(set([dirname(x) for x in dirs_process]))
        dirs_process.sort()


        ligfiles_dict = {}
        for dir in dirs_process:
            ligfiles = []
            [ligfiles.extend(glob.glob(os.path.join(dir, f'{pat}'))) for pat in args.lig_patterns]
            ligfiles = [x for x in ligfiles if not any([pat in basename(x) for pat in LIG_PATTERNS_EXCLUDE])]
            ligfiles_dict[dir] = ligfiles

        print(f'\nFound {len(dirs_process)} directories to process:')
        if args.verbose or True:
            for dir, ligfiles in ligfiles_dict.items():
                print(dir, end='\n    ')
                print('\n    '.join([x.replace(dir, '') for x in ligfiles]))
        
        # get approval from user
        inp = '' if not args.yes else 'y'
        while inp not in ['y','n']:
            inp = input(f'\nProcess the above {len(dirs_process)} directories? (y/n):  ').lower()
        if inp == 'n':
            sys.exit('Aborting...')


        pbar = tqdm(ligfiles_dict.items())
        for dir, ligfiles in pbar:

            if len(ligfiles) == 0:
                warnings.warn(f'\nCould not find any ligand files using patterns "{args.lig_patterns}" in path "{dir}". Skipping...')
                continue
            
            recfile = args.recfile
            desc = f'Processing'
            if recfile is None:
                # find receptor file based on pdb_id
                pdb_id = get_pdb_from_path(dir)
                if pdb_id is None or pdb_id == '':
                    pdb_id = get_pdb_from_path(dir, args.rec_dirs)            
                desc = f'Processing {pdb_id}'
                pbar.set_description(desc)
                for rec_dir in args.rec_dirs:
                    # print('pattern search:', os.path.join(rec_dir, f'**/{pdb_id}_pocket.pdb'))
                    # recfiles = glob.glob(os.path.join(rec_dir, f'**/{pdb_id}_pocket.pdb'), recursive=True)
                    # recfiles = glob.glob(os.path.join(rec_dir, f'**/{pdb_id}/'), recursive=True)
                    if pdb_id in os.listdir(rec_dir):
                        recfile = glob.glob(join(rec_dir, pdb_id, f'{pdb_id}*_pocket.pdb'))
                        recfile = recfile[0] if len(recfile) > 0 else None
                        if recfile is None:
                            recfile = glob.glob(join(rec_dir, pdb_id, f'{pdb_id}*_protein.pdb'))
                            recfile = recfile[0] if len(recfile) > 0 else None
                        if recfile is not None:
                            if args.verbose:
                                print(f'Found receptor_file: {recfile}', flush=True)
                            break

            if recfile is None:
                warnings.warn(f'Could not find any receptor file using pattern "**/{pdb_id}*_pocket.pdb" in paths "{args.rec_dirs}". Skipping...')
                continue
            
            # print('processing files...', flush=True)
            for i in range(len(ligfiles)):
                pbar.set_description(desc + f' ({i}/{len(ligfiles)})')
                ligfile = ligfiles[i]
                qvina_process_multi(ligfile, recfile, 
                                    qvina_cmd=QVINA_PATH, 
                                    outfile=None, 
                                    num_procs=8, 
                                    score=args.score, 
                                    score_only=args.score_only, 
                                    protonate=args.protonate,  
                                    exhaustiveness=args.exhaustiveness, 
                                    verbose=args.verbose, 
                                    pbar=args.pbar, 
                                    overwrite=args.overwrite)

        t1 = time.time() - t0
        print('\n\nTotal time: %dm %ds'%(t1//60, t1%60))
            
            
            
            

    




