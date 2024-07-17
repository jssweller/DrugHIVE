
import os, sys, shutil, glob
import gzip
import h5py
import torch
import subprocess, logging
from typing import Union
from functools import partial
from collections import defaultdict, Counter
from typing import Iterable
from copy import copy
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, RDLogger, DataStructs, RDConfig
from rdkit.Chem import AllChem, Lipinski, rdFMCS
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.MolSurf import TPSA
from rdkit.Geometry import Point3D

if os.path.join(RDConfig.RDContribDir) not in sys.path:
    sys.path.append(os.path.join(RDConfig.RDContribDir))
from SA_Score import sascorer

from .gridutils import rot3d, trans3d

ATOMIC_NUMS_DEFAULT = [6,7,8,9,15,16,17,35,53] 

def get_mol_center(mol):
    '''Calculates the center of an rdkit molecule.'''
    return mol.GetConformer().GetPositions().mean(axis=0)

def shift_mol(mol, shift_vec):
    '''Translates mol coordinates by input vector.'''
    mol = Mol2(mol)
    positions = mol.get_coords()
    positions = np.array(positions) + shift_vec
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = positions[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    return Chem.Mol(mol)

def recenter_mol(mol, new_center):
    '''Re-centers coordinates of mol to new_center.'''
    if not isinstance(new_center, np.ndarray):
        new_center = np.asarray(new_center, dtype=float)
    mc = get_mol_center(mol)
    mol = shift_mol(mol, -mc + new_center)
    return mol

def recenter_mol_from_ref(mol, mol_ref):
    '''Re-centers mol to match center of mol_ref.'''
    return recenter_mol(mol, get_mol_center(mol_ref))

def rotate_coordinates(mol, angles, inverse=False):
    coords = mol.GetConformer().GetPositions()
    coords = rot3d(coords, angles, inverse)
    update_coordinates(mol, coords)

def translate_coordinates(mol, vec, inverse=False):
    coords = mol.GetConformer().GetPositions()
    coords = trans3d(coords, vec, inverse)
    update_coordinates(mol, coords)

def transform_coordinates(mol, trans_matrix):
    coords = mol.GetConformer().GetPositions()
    coords_aligned = coords @ trans_matrix
    update_coordinates(mol, coords_aligned)

def update_coordinates(mol, coords):
    '''Updates a rdkit molecule's coordinates.'''
    coords = coords.reshape(-1,3)
    conf = mol.GetConformer()
    i = 0
    for _ in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        x,y,z = tuple(coords[i])
        conf.SetAtomPosition(i,Point3D(float(x), float(y), float(z)))
        i += 1


def get_all_frags(mol : Chem.Mol):
    '''Separates disconnected fragments of molecule into individual molecules.'''
    frags_all = []
    mol2 = copy(mol)
    frag, mol2 = get_largest_fragment(mol2, return_frags=True)
    if frag.GetNumAtoms() > 0:
        frags_all.append(frag)
    while mol2.GetNumAtoms() > 0:
        frag, mol2 = get_largest_fragment(mol2, return_frags=True)
        if frag.GetNumAtoms() > 0:
            frags_all.append(frag)
    return frags_all


def get_largest_fragment(mol : Chem.Mol, return_frags : bool = False, verbose : bool = False):
    """Separates and returns largest fragment in molecule. Fragment is defined as largest connected substructure. If return_frags is true, returns remaining fragments as a separate Chem.Mol
    """    
    mol_clean = copy(mol)

    smi = Chem.MolToSmiles(mol_clean, canonical=False) 
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

def protonate_mol_sdf(mol, ph=7.2):
    '''Protonates molecule with openbabel.'''
    ftmp = '/tmp/tmp_prot.sdf'
    fout = ftmp.replace('.sdf', '.p.sdf')
    write_mols_sdf(mol, ftmp)
    cmd = f'obabel {ftmp} -O {fout} -p {ph}'
    cmd_return = subprocess.run(cmd, capture_output=True, shell=True)
    assert cmd_return.returncode == 0, f'Protonation with OpenBabel failed\nstdout: {cmd_return.std_out}\nstderr: {cmd_return.std_error}'
    return MolParser(fout).get_rdmol(sanitize=False)

       
def is_3d_molblock(moltext):
    '''Checks if mol block is 3D.'''
    lines = moltext.strip().split('\n')
    for line in lines:
        if 'RDKit' in line:
            dim = line.split()[1]
            return dim == '3D'
    raise Exception('molecule dimensions not found!')


def is_3d_mol(mol):
    '''Checks if mol has 3D coordinates.'''
    if len(mol.GetConformers()) == 0:
        return False
    else:
        conf = mol.GetConformer()
        if (conf.GetPositions() == 0).all(axis=0).any():
            return False
    return True

def molname_from_molblock(molblock):
    '''Gets the ZINC name from molecule file text.'''
    molname = ''
    lines = molblock.strip().split('\n')
    for line in lines:
        if 'ZINC' in line:
            molname = line.strip()
    return molname

def removeHeteroAtoms(mol):
    '''Removes atoms with label HETATM in pdb file. These are H2O, ions, etc.'''
    atoms_remove = []
    for a in mol.GetAtoms():
        inf = a.GetMonomerInfo()
        if inf.GetIsHeteroAtom():
            atoms_remove.append(a.GetIdx())
    mol = Chem.EditableMol(mol)

    for i in reversed(sorted(atoms_remove)):
        mol.RemoveAtom(i)
    return mol.GetMol()

def removeWaters(mol):
    '''Removes atoms with Residue Name "HOH" in pdb file. These are H2O, ions, etc.'''
    atoms_remove = []
    for a in mol.GetAtoms():
        inf = a.GetMonomerInfo()
        if inf.GetResidueName() == 'HOH':
            atoms_remove.append(a.GetIdx())
    mol = Chem.EditableMol(mol)

    for i in reversed(sorted(atoms_remove)):
        mol.RemoveAtom(i)
    return mol.GetMol()


def calc_avg_diversity(mols, fp_type='rdkit'):
    '''Calculates the average diversity (1 - similarity) of a set of molecules.'''
    return 1 - calc_avg_similarity(mols, fp_type)


def calc_avg_similarity(mols, fp_type='rdkit'):
    '''Calculates the average similarty of a set of molecules.'''
    if len(mols) < 2:
        return 0.
    
    if fp_type == 'morgan':
        fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in mols]
    elif fp_type == 'rdkit':
        fps = [FingerprintMol(x) for x in mols]
    
    div_total = 0
    count = 0
    for i in range(len(mols)-1):
        sims = calc_similarity(mols[i+1:], mols[i])
        count += len(sims)
        div_total += sims.sum()

    return div_total / count


def calc_similarity(mols, mol_ref, fp_type='rdkit'):
    '''Calculates the Tanimoto Similarity to a reference molecule for each of a set of mols.'''
    if not isinstance(mols, list):
        mols = [mols]
        
    [Chem.GetSSSR(m) for m in mols]
    Chem.GetSSSR(mol_ref)
    
    if fp_type == 'morgan':
        fps_probes = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in mols]
        fps_ref = AllChem.GetMorganFingerprintAsBitVect(mol_ref,2,1024)
    elif fp_type == 'rdkit':
        fps_probes = [FingerprintMol(x) for x in mols]
        fps_ref = FingerprintMol(mol_ref)
        
    sims = np.array([DataStructs.TanimotoSimilarity(x, fps_ref) for x in fps_probes])
    
    if mol_ref.GetNumAtoms() == 1:
        sims[:] = float('nan')
    for i in range(len(sims)):
        if mols[i].GetNumAtoms() == 1:
            sims[i] = float('nan')

    return sims

def hba_qed(mol):
    '''Gets number of hydrogen bond acceptors in a molecule as defined in the Chem.QED module.'''
    Acceptors = Chem.QED.Acceptors
    return  sum(len(mol.GetSubstructMatches(pattern)) for pattern in Acceptors
                if mol.HasSubstructMatch(pattern))

def arom_qed(mol):
    '''Gets number of aromatic rings in a molecule as defined in the Chem.QED module.'''
    AliphaticRings = Chem.QED.AliphaticRings
    arom = Chem.GetSSSR(Chem.DeleteSubstructs(Chem.Mol(mol), AliphaticRings))
    if not isinstance(arom, int):
        arom = len(arom)
    return arom

def rotb_qed(mol):
    '''Gets number of rotatable bonds in a molecule as defined in the Chem.QED module.'''
    return Chem.rdMolDescriptors.CalcNumRotatableBonds(mol, Chem.rdMolDescriptors.NumRotatableBondsOptions.Strict)


DEFAULT_PROPERTY_FUNCS = {'hba': hba_qed,
                          'hbd': AllChem.CalcNumHBD,
                          'logp': MolLogP,
                          'alogp': MolLogP,
                          'rotb': rotb_qed,
                          'arom' : arom_qed,
                          'psa' : TPSA,
                          'mw' : Chem.rdMolDescriptors._CalcMolWt,
                          'width' : lambda m : Chem.rdmolops.Get3DDistanceMatrix(m).max(),
                          'natoms' : Chem.rdMolDescriptors.CalcNumHeavyAtoms,
}

def get_mol_stats(mols, keys=None, keys_exclude=None, include_morgan=False):
    '''Calculates the properties of a molecule or set of molecules.'''
    if not isinstance(mols, list):
        mols = [mols]
        
    allprops = defaultdict(list)
    
    qedprops = ['MW',   # molecular weight
                'ALOGP', # crippen logp (solubility), octanol–water partition coefficient
                'PSA',   # polar surface area
                'ROTB',  # number of rotatable bonds
                'AROM',  # number of aromatic rings
                'HBA',   # number of H-bond acceptors
                'HBD',   # number of H-bond donors
                ]  
    
    atomic_nums = ATOMIC_NUMS_DEFAULT  
    
    def calc_num_atoms_each(mol):
        count = Counter([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        dat = np.array([count[anum] for anum in atomic_nums], dtype=np.int16)
        return dat    

    def qed_alerts(mol):
        return sum(1 for alert in Chem.QED.StructuralAlerts if mol.HasSubstructMatch(alert))
    

    def err_wrapper(func):
        def wrap(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return np.nan
        return wrap
    
    propfuncs = {
                 'hba_lipinski': Chem.rdMolDescriptors.CalcNumLipinskiHBA,
                 'hbd_lipinski': Chem.rdMolDescriptors.CalcNumLipinskiHBD,
                 'num_rings' : lambda m : len(m.GetRingInfo().AtomRings()),
                 'ring_sizes' : lambda m : [len(x) for x in m.GetRingInfo().AtomRings()],
                #  'ring_sizes' : lambda m : [len(x) for x in Chem.GetSSSR(m)],
                 'num_atoms' : Chem.rdMolDescriptors.CalcNumHeavyAtoms,
                 'num_het_atoms' : Chem.rdMolDescriptors.CalcNumHeteroatoms,
                 'num_atoms_each' : calc_num_atoms_each,
                #  'num_stereo_centers': Chem.rdMolDescriptors.CalcNumAtomStereoCenters,
                 'mol_width' : lambda m : Chem.rdmolops.Get3DDistanceMatrix(m).max(),
                 'npr1': Chem.rdMolDescriptors.CalcNPR1,
                 'npr2': Chem.rdMolDescriptors.CalcNPR2,
                 'sa' : sascorer.calculateScore,
                 'murcko' : lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m),
                 'morgan_vec' : lambda m : np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024), dtype=bool),
                 'alogp': Chem.Crippen.MolLogP,
                 'hba' : hba_qed, # AllChem.CalcNumHBA,
                 'hbd' : AllChem.CalcNumHBD,
                 'rotb': lambda m: Chem.rdMolDescriptors.CalcNumRotatableBonds(m, Chem.rdMolDescriptors.NumRotatableBondsOptions.Strict),
                 'arom' : arom_qed, # Chem.rdMolDescriptors.CalcNumAromaticRings,
                 'psa' : Chem.MolSurf.TPSA,
                 'mw' : Chem.rdMolDescriptors._CalcMolWt,
                 'qed': None,
                }
        
    if not include_morgan:
        del propfuncs['morgan_vec']
    
    
    if keys_exclude is not None:
        assert keys is None, 'Provide either input `keys` or `keys_exclude`, not both.'
        keys = [k for k in propfuncs.keys() if k not in keys_exclude]
    
    if keys is None:
        keys = list(propfuncs.keys())

    
    if 'qed' in keys and len(keys) > 1:
        keys.append(keys.pop(keys.index('qed'))) # make sure 'qed' is last
    props_include = keys

    for m in mols:
        if m is not None:
            m = Chem.RemoveHs(m, updateExplicitCount=True, sanitize=False)
            flag = Chem.SanitizeMol(m, catchErrors=True)

            for p in props_include:
                func = err_wrapper(propfuncs[p])
                if p == 'qed':
                    props_dict = {}
                    for key in qedprops:
                        if not key.lower() in props_include:
                            props_dict[key] = err_wrapper(propfuncs[key.lower()])(m)
                        else:
                            props_dict[key] = allprops[key.lower()][-1]
                    try:
                        allprops[p].append(Chem.QED.qed(m, qedProperties=Chem.QED.QEDproperties(**props_dict, ALERTS=qed_alerts(m))))
                    except:
                        allprops[p].append(np.nan)
                else:
                    try:
                        allprops[p].append(func(m))
                    except Exception as e:
                        print(p, func)
                        raise e
        else:
            for p in props_include:
                allprops[p].append(np.nan)

    if 'num_atoms_each' in allprops.keys():
        vals = np.stack(allprops['num_atoms_each'])
        for i, anum in enumerate(atomic_nums):
            allprops[f'e{anum}'] = vals[:,i]
        del allprops['num_atoms_each']

    for p in allprops.keys():
        if not any([p in x for x in ['ring_sizes', 'morgan_vec']]):
            allprops[p] = np.array(allprops[p])
    
    allprops = pd.DataFrame.from_dict(allprops)
    if 'sa' in allprops.columns:
        allprops['sa'] = np.round((10 - allprops['sa']) / 9, 2)  # from pocket2mol paper
    return allprops

def align_mols(mols, mol_ref=None, verbose=False):
    '''Aligns each of a list of mols to either a reference molecule or to the first molecule in the list.'''
    if not isinstance(mols, list):
        mols = [mols]
        
    if mol_ref is None:
        mol_ref = mols[0]
        mols_align = mols[1:]
    else:
        mols_align = mols
        
    mcs = rdFMCS.FindMCS(mols_align+[mol_ref])
    if verbose:
        print('Num atoms in MCS:', mcs.numAtoms)
    if mcs.numAtoms < 3:
        print('Alignment failed. No common substructure.')
        return mols
    
    patt = Chem.MolFromSmarts(mcs.smartsString)
    ref_sub_atoms = mol_ref.GetSubstructMatch(patt)
    rmsds = []
    transforms = []
    for mol in mols_align:
        mol_sub_atoms = mol.GetSubstructMatch(patt)
        rmsd, transform = Chem.rdMolAlign.GetAlignmentTransform(mol, mol_ref, maxIters=len(mol.GetAtoms()), atomMap=list(zip(mol_sub_atoms, ref_sub_atoms)))
        Chem.rdMolTransforms.TransformConformer(mol.GetConformer(), transform)
#         rmsd = Chem.rdMolAlign.AlignMol(mol, mol_ref, maxIters=len(mol.GetAtoms()), atomMap=list(zip(mol_sub_atoms, ref_sub_atoms)))
        rmsds.append(rmsd)
        transforms.append(transform)
    return rmsds, transforms


def get_lig_pocket(lig, prot, distance_thresh=8):
    '''Extracts a protein pocket based on distance cutoff from a ligand.'''
    if distance_thresh > 0:
        lig_coords = lig.GetConformer().GetPositions()
        prot_coords = prot.GetConformer().GetPositions()
        dists = cdist(prot_coords, lig_coords).min(axis=1)

        prot_mask = dists < distance_thresh
        rec_atoms_keep = np.nonzero(prot_mask)[0]

        rec_residues_keep = set()
        for i in rec_atoms_keep.tolist():
            atom = prot.GetAtomWithIdx(i)
            pdb_inf = atom.GetPDBResidueInfo()
            rec_residues_keep.add((pdb_inf.GetChainId(), pdb_inf.GetResidueNumber()))

        rec = Chem.RWMol(prot)
        for i in reversed(range(rec.GetNumAtoms())):
            atom = rec.GetAtomWithIdx(i)
            pdb_inf = atom.GetPDBResidueInfo()
            chain_res = (pdb_inf.GetChainId(), pdb_inf.GetResidueNumber())
            if chain_res not in rec_residues_keep:
                rec.RemoveAtom(i)
    else:
        rec = Chem.Mol(prot)
    return prot


def get_mol_energy(lig, remove_hs=True, mmff=False):
    ''''Returns the energy of a molecule using either UFF or MMFF force field.'''
    reason = ''
    if lig is None or lig.GetNumHeavyAtoms() < 2:
        reason = 'MOL_NONE'
        return float('nan')
    
    if remove_hs:
        lig = Chem.RemoveHs(lig, sanitize=False)
        
    try:
        Chem.SanitizeMol(lig)
    except Exception as e:
        if not isinstance(e, Chem.KekulizeException):
            # print('Molecule Sanitization failed. Returning `None`.')
            reason = 'SANITIZE_FAIL'
            return float('nan')


    Chem.GetSSSR(lig)
    lig.UpdatePropertyCache(strict=False)
    
    if not mmff:
        ff = AllChem.UFFGetMoleculeForceField(lig, confId=0, ignoreInterfragInteractions=False)
    else:
        props = AllChem.MMFFGetMoleculeProperties(lig)
        ff = AllChem.MMFFGetMoleculeForceField(lig, props, confId=0, ignoreInterfragInteractions=False)
    if ff is None:
        return float('nan')
    ff.Initialize()
    e0 = ff.CalcEnergy()
    return e0


def get_ff(mol, mmff=False):
    '''Gets the rdkit force field for a molecule. Either MMFF or UFF.'''
    if not mmff:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=0, ignoreInterfragInteractions=False)
    else:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0, ignoreInterfragInteractions=False)
    return ff



def ff_optimize_mol(mol, n_tries=1, align=True, mmff=False, max_attempts=5, max_iters=200, remove_hs=True, atoms_fix=None):
    '''Optimizes a molecule a number of times using either UFF or MMFF force field in rdkit.'''
    ebest = 1e20
    returnval = None
    mol0 = copy(mol)

    for i in range(n_tries):
        molopt = copy(mol0)
        its = max_iters + np.random.randint(50)
        molopt, e0, e1, success, reason = _ff_optimize_mol(mol, align=align, mmff=mmff, max_attempts=max_attempts, max_iters=its, remove_hs=remove_hs, atoms_fix=atoms_fix)
        if success and e1 < ebest:
            ebest = e1
            returnval = (molopt, e0, e1, success, reason)
    if returnval is None:
        returnval = None, float('nan'), float('nan'), False, reason
    return returnval



def _ff_optimize_mol(mol, align=True, mmff=False, max_attempts=5, max_iters=200, remove_hs=True, atoms_fix=None):
    '''Optimizes a molecule using either UFF or MMFF force field in rdkit.'''
    mol = copy(mol)
    reason = ''
    if mol is None or mol.GetNumHeavyAtoms() < 2:
        reason = 'MOL_NONE'
        return None, float('nan'), float('nan'), False, reason
    
    mol = Chem.RemoveHs(mol, sanitize=False)
    Chem.GetSSSR(mol)
    molopt = Chem.Mol(copy(mol))
    try:
        Chem.SanitizeMol(molopt)
    except Exception as e:
        if not isinstance(e, Chem.KekulizeException):
            # print('Molecule Sanitization failed. Returning `None`.')
            reason = 'SANITIZE_FAIL'
            return None, float('nan'), float('nan'), False, reason

    try:
        n_attempts = 0
        success = False
        while not success and (n_attempts < max_attempts):
            molopt = Chem.AddHs(Chem.RemoveHs(molopt), addCoords=True)
            ff = get_ff(molopt, mmff)
            if ff is None:
                # print('Couldn\'t get FF. Returning `None`.')
                reason = 'FF_FAIL'
                return None, float('nan'), float('nan'), False, reason
            ff.Initialize()
            if n_attempts == 0:
                e0 = ff.CalcEnergy()

            if atoms_fix is not None:
                for ai in atoms_fix:
                    ff.AddFixedPoint(ai)
            success = ff.Minimize(maxIts=max_iters, forceTol=0.0001) == 0
            n_attempts += 1
    except Exception as e:
        # print('Optimization failed. Returning `None`.')
        msg = e.with_traceback(e.__traceback__)
        success = False
        reason = 'MINIMIZE_FAIL'
        print(e)
        return None, float('nan'), float('nan'), success, reason

    e1 = ff.CalcEnergy()

    if remove_hs:
        molopt = Chem.RemoveHs(molopt)
    if align:
        try:
            idxs0 = [i for i,a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() != 1]
            idxs1 = [i for i,a in enumerate(molopt.GetAtoms()) if a.GetAtomicNum() != 1]
            atom_map=list(zip(idxs1, idxs0))
            Chem.rdMolAlign.AlignMol(molopt, mol, maxIters=len(molopt.GetAtoms()), atomMap=atom_map)
        except Exception as e:
            # print('\nAlignment failed. Returning unaligned mol.')
            reason = 'ALIGN_FAIL'

    if not success:
        reason = 'NOT_CONVERGED'
    
    return molopt, e0, e1, success, reason


def optimize_mols_multi(mols, max_iters=200, max_attempts=5, num_procs=10, **kwargs):
    '''Optimizes a list of molecules using either UFF or MMFF force field in rdkit.'''
    RDLogger.DisableLog('rdApp.*')
    results = []

    num_procs = min(num_procs, len(mols))
    if num_procs == 1:
        result = [ff_optimize_mol(mols[0], max_iters=max_iters)]
    else:
        with Pool(processes=num_procs) as pool:
            result = pool.map(partial(ff_optimize_mol, max_iters=max_iters, max_attempts=max_attempts, **kwargs), mols)   
    results.extend(list(result))

    mols_opt = [x[0] for x in results]
    energy_init = [x[1] for x in results]
    energy_final = [x[2] for x in results]
    success_bools = [x[3] for x in results]
    reasons = [x[4] for x in results]
    return mols_opt, energy_init, energy_final, success_bools, reasons


def write_mols_sdf(mols, file, remove_hs=True, append=False):
    '''Writes a list of rdkit molecules to a single .sdf file.'''
    if isinstance(mols, (Chem.Mol, Mol2)):
        mols = [mols]
    if not isinstance(mols, list) and hasattr(mols, 'tolist'):
        mols = mols.tolist()
    
    os.makedirs(os.path.dirname(file), exist_ok=True)
    mode = 'w+' if not append else 'a+'
    with open(file, mode=mode) as f, Chem.rdmolfiles.SDWriter(f) as writer:
        writer.SetKekulize(False)
        for mol in mols:
            if mol is None:
                mol = Chem.MolFromSmiles('C')  # dummy mol with single carbon
            if remove_hs:
                mol = Chem.RemoveHs(mol, updateExplicitCount=True, sanitize=False)
            try:
                writer.write(mol)
            except Exception as e:
                if isinstance(e, RuntimeError):
                    print('writing mol failed. Writing dummy mol.')
                    mol = Chem.MolFromSmiles('C')
                    writer.write(mol)
                else:
                    raise e


class AtomFeaturizer(object):
    def __init__(self):
        '''Class for calculating the features of atoms in rdkit molecules.'''
        self.feat_names = ['HBD', 'HBA', 'Aromatic', 'FChargePos', 'FChargeNeut', 'FChargeNeg']

    def get_hba(self, mol):
        dat = mol.GetSubstructMatches(Lipinski.HAcceptorSmarts, uniquify=1)
        if len(dat) == 0:
            dat = []
        else:
            dat = np.concatenate(dat).tolist()
        return {'HBA': dat}
    
    def get_hbd(self, mol):
        dat = mol.GetSubstructMatches(Lipinski.HDonorSmarts, uniquify=1)
        if len(dat) == 0:
            dat = []
        else:
            dat = np.concatenate(dat).tolist()
        return {'HBD': dat}

    def get_formal_charges(self, mol):
        feat_dict = {'FChargePos':[],'FChargeNeut':[],'FChargeNeg':[]}
        for atom in mol.GetAtoms():
            c = atom.GetFormalCharge()
            if c > 0:
                feat_dict['FChargePos'].append(atom.GetIdx())
            elif c < 0:
                feat_dict['FChargeNeg'].append(atom.GetIdx())
            else:
                feat_dict['FChargeNeut'].append(atom.GetIdx())
        return feat_dict


    def get_aromatic(self, mol):
        feat_dict = {'Aromatic': []}
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                feat_dict['Aromatic'].append(atom.GetIdx())
        return feat_dict


    def get_features(self, mol):
        mol.UpdatePropertyCache(strict=False)
        fdict = self.get_aromatic(mol)
        fdict.update(self.get_hba(mol))
        fdict.update(self.get_hbd(mol))
        fdict.update(self.get_formal_charges(mol))
        return fdict



class Mol2(Chem.rdchem.Mol):
    '''Wrapper class for rdkit molecules with added convienence methods.'''
    def get_atom(self, i):
        return self.GetAtomWithIdx(i)

    def get_dist_matrix(self):
        return Chem.rdmolops.Get3DDistanceMatrix(self)

    def get_bond_matrix(self, include_order=True):
        return Chem.rdmolops.GetAdjacencyMatrix(self, useBO=include_order)

    def get_bonds_sparse(self, include_order=True):
        bmat = self.get_bond_matrix(include_order)
        smat = np.zeros((int((np.triu(bmat) > 0).sum()), 3), dtype=np.float32)
        k = 0
        for i in range(bmat.shape[0]):
            for j in range(i, bmat.shape[1]):
                if bmat[i, j] > 0:
                    btype = bmat[i, j]
                    smat[k] = [i, j, btype]
                    k += 1
        return smat

    def get_coords(self, canonical=False):
        if canonical:
            return self.GetConformer().GetPositions() - self.get_center()
        else:
            return self.GetConformer().GetPositions()

    def get_atom_nums(self):
        types = []
        for i, a in enumerate(self.GetAtoms()):
            types.append(a.GetAtomicNum())
        return np.array(types, dtype=int)

    def get_centroid(self):
        '''returns the mean of the atom coordinates in each dimension.'''
        return Chem.rdMolTransforms.ComputeCentroid(self.GetConformer())

    def get_center(self):
        '''returns the center of the atom coordinate range in each dimension.'''
        coords = self.GetConformer().GetPositions()
        return (coords.max(axis=0) + coords.min(axis=0))/2

    def get_num_rings(self):
        return len(self.GetRingInfo().AtomRings())

    def get_zinc_id(self):
        return self.GetProp('zinc_id')

    def get_smiles(self):
        return Chem.rdmolfiles.MolToSmiles(self)

    def canonicalize(self):
        Chem.rdMolTransforms.CanonicalizeMol(self)

    def plot(self, **kwargs):
        return rdkit.Chem.Draw.IPythonConsole.drawMol3D(self, **kwargs)

    def canonicalize_mols(mols):
        for m in mols:
            m.canonicalize()

    def get_extended_mols(mols):
        return [Mol2(mol) for mol in mols]

    def get_all_types(mols):
        return [m.get_atom_nums() for m in mols]

    def get_all_coords(mols, canonical=False):
        return [m.get_coords(canonical) for m in mols]

    def get_all_dist_matrices(mols):
        return [m.get_dist_matrix() for m in mols]

    def get_all_numrings(mols):
        return [m.get_num_rings() for m in mols]

    def get_all_names(mols):
        return [m.get_zinc_id() for m in mols]



class MolParser():
    def __init__(self,
                 fname=None,
                 savedir=None,
                 atomtyper=None):
        '''Class for parsing and loading .sdf and .pdb files into rdkit molecules.'''

        self.fname = fname
        self.savedir = savedir
        self.file = None

        if atomtyper is None:
            atomtyper = AtomTyperDefault()
        self.atomtyper = atomtyper
        self.atom_featurizer = AtomFeaturizer()
        
        if fname is not None:
            self.open(fname)

    @property
    def fname(self):
        return self._fname
    
    @fname.setter
    def fname(self, val):
        self._fname = val
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next_rdmol()

    def close(self):
        self.file.close()

    def open(self, fname):
        if self.check_open():
            self.close()
        if fname is not None:
            self.fname = fname
        if self.fname.endswith('.gz'):
            self.file = gzip.open(self.fname, 'rb')
        else:
            self.file = open(self.fname, 'r')
        self.count = 0

    def check_open(self):
        if self.file is None:
            return False
        else:
            return (not self.file.closed)

    def atom_symbols_to_nums(self, symbols):
        nums = []
        for s in symbols:
            if '.' in s:
                s = s.split('.')[0]
            nums.append(self.ptable.GetAtomicNumber(s))
        return np.array(nums, dtype=int)

    def types_list2vec(self, types_list):
        types_len = max([len(x) for x in types_list])
        types = np.zeros((len(types_list),types_len), dtype=int) - 1
        for ti, tlist in enumerate(types_list):
            types[ti,:len(tlist)] = tlist
        return types
    

    def mol_data_from_file(self, molfile):
        self.fname = molfile
        mol = self.get_rdmol(sanitize=False, removeHs=False)
        Chem.GetSSSR(mol)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY, catchErrors=True)

        mol = Mol2(mol)
        atom_types = self.get_atom_types(mol)
        atom_types = self.types_list2vec(atom_types)
        atom_coords = mol.get_coords()

        idxs = atom_types[:,0] != 1
        atom_types = atom_types[idxs]
        atom_coords = atom_coords[idxs]
        return atom_coords, atom_types


    def get_bonds_types(self, rdmol2):
        bmat : np.ndarray = rdmol2.get_bonds_sparse(include_order=True)
        types_list = []
        for i in range(len(bmat)):
            types_list.append([])
            
        for i in range(len(bmat)):
            b_order = bmat[i,2]
            t = self.atomtyper.bond_order2type(b_order)
            t = self.atomtyper.type2num(t)
            types_list[i].append(t)
        return bmat, types_list
            
    
    def get_atom_types(self, mol, max_width=10):
        atom_nums = np.array(mol.get_atom_nums(), dtype=int)
        types_list = []
        for i in range(len(atom_nums)):
            types_list.append([])
        
        for i in range(len(atom_nums)):
            t = 'e'+str(atom_nums[i])
            tnum = self.atomtyper.type2num(t)
            types_list[i].append(tnum)

        atom_feats = self.atom_featurizer.get_features(mol)
        for feat_name, atom_idxs in atom_feats.items():
            tnum = self.atomtyper.type2num(feat_name)
            for i in atom_idxs:
                types_list[i].append(tnum)

        return types_list


    def get_rdmol(self, sanitize=True, remove_hetero=True, removeHs=True, remove_water=False):
        if self.fname.endswith('.mol2') or self.fname.endswith('.mol2.gz'):        
            read_func = partial(Chem.rdmolfiles.MolFromMol2File, cleanupSubstructures=False)
        elif self.fname.endswith('.pdb') or self.fname.endswith('.bio1'):
            read_func = Chem.rdmolfiles.MolFromPDBFile
        elif self.fname.endswith('.sdf'):
            read_func = Chem.rdmolfiles.SDMolSupplier
        
        remove_water &= not remove_hetero

        rdmol = read_func(self.fname, sanitize=sanitize, removeHs=False)

        if read_func == Chem.rdmolfiles.SDMolSupplier:
            rdmol = next(rdmol)
        
        if rdmol is None:
            print('failed to read molecule.\tTrying without sanitize...')
            rdmol = read_func(self.fname, sanitize=False)
            if read_func == Chem.rdmolfiles.SDMolSupplier:
                rdmol = next(rdmol)
            if rdmol is None:
                print('Skipping mol\n',self.fname)
                return rdmol
        if removeHs:
            try:
                rdmol = Chem.RemoveHs(rdmol, updateExplicitCount=True, sanitize=False)
            except:
                print('mol\n',self.fname)
                raise Exception

        if remove_hetero and self.fname.endswith('.pdb'):
            rdmol = removeHeteroAtoms(rdmol)
        elif remove_water and self.fname.endswith('.pdb'):
            rdmol = removeWaters(rdmol)

        rdmol = Mol2(rdmol)
        return rdmol
    
    def get_next_rdmol(self):
        self.get_rdmol()


class PDBParser(MolParser):
    '''Class for parsing and extracting data from .pdb files.'''

    def save_coords_types(self,
                        file,
                        savedir,
                        numstart=0,
                        chunks=1000,
                        numread='all',
                        mol_natoms_max=40,
                        center_mols=True,
                        compression=None,
                        verbose=True,
                        debug=False,
                        **kwargs
                        ):

        

        self.fname = file
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        readall = (numread == 'all')
        
        mol_bonds_max = int(2*mol_natoms_max)
        types_atoms_max = 10
        types_bonds_max = 10
        

        if readall:
            nmols = self.count_mols(numstart=numstart)
        else:
            nmols = numread
        
        chunks = min(chunks, nmols)
        print('nmols', nmols)
        self.open()
        with h5py.File(os.path.join(savedir, os.path.basename(file).replace('.mol2.gz','.h5')), 'w') as f5:
            dat_atom_types = f5.create_dataset("atom_types", 
                                          shape=(nmols, mol_natoms_max, types_atoms_max), 
                                          chunks=(chunks, mol_natoms_max, types_atoms_max), 
                                          compression=compression,
                                          dtype='int16',
                                          fillvalue=-1,
                                          **kwargs
                                               
                                          )

            dat_coords = f5.create_dataset("atom_coords", 
                                           shape=(nmols, mol_natoms_max, 3), 
                                           chunks=(chunks, mol_natoms_max, 3), 
                                           compression=compression,
                                           dtype='float32', 
                                           fillvalue=np.nan,
                                          **kwargs
                                          )

            dat_bonds = f5.create_dataset("bonds", 
                                           shape=(nmols, mol_bonds_max, 3), 
                                           chunks=(chunks, mol_bonds_max, 3), 
                                           compression=compression,
                                           dtype='float32', 
                                           fillvalue=0,
                                          **kwargs
                                         )
            
            dat_bond_types = f5.create_dataset("bond_types", 
                                          shape=(nmols, mol_bonds_max, types_bonds_max), 
                                          chunks=(chunks, mol_bonds_max, types_bonds_max), 
                                          compression=compression,
                                          dtype='int16',
                                          fillvalue=-1,
                                          **kwargs
                                              )
                       
            dt = h5py.special_dtype(vlen=str)
            dat_names = f5.create_dataset("names", 
                                           shape=(nmols), 
                                           chunks=(chunks), 
                                           compression=compression,
                                           dtype=dt,
                                          **kwargs
                                         )


            if numstart > 0:
                print('Starting at molecule: %d' % numstart)
                seekpos = self.skip_to_mol(molnum=numstart)
            
            i = 0
            while True:
                try:
                    moldata = self.get_next_data_rdkit(canonicalize=center_mols)
                except:
                    print('i',i, '   count:',self.count)
                    raise Exception
                n_atoms = len(moldata['atom_types'])
                
                dat_names[i] = moldata['names']
                dat_coords[i, :n_atoms] = moldata['coords']
                dat_bonds[i, :len(moldata['bonds'])] = moldata['bonds']                

                for ia in range(len(moldata['bond_types'])):
                    t = moldata['bond_types'][ia]
                    dat_bond_types[i,ia, :len(t)] = t
                for ia in range(len(moldata['atom_types'])):
                    t = moldata['atom_types'][ia]
                    dat_atom_types[i, ia, :len(t)] = t
                
                i += 1
                if debug:
                    print('mol:', self.count)
                    
                if verbose and ((i+1) % max(chunks,1000) == 0):
                    print('molecules %dk processed' % (self.count//1e3))
                if self.endfile:
                    break
                if not readall:
                    if i >= numread:
                        break
        self.file.close()
        return self.count-numstart


class BulkMol2Parser(MolParser):
    '''Class for parsing, extracting data from, and loading rdkit molecules from multi-molecule .sdf and .pdb files.'''
    def __init__(self,
                 fname=None,
                 savedir=None,
                 atomtyper=None,
                 mol_pattern=b'@<TRIPOS>MOLECULE',
                 name_pattern=b'ZINC',
                 atom_pattern=b'@<TRIPOS>ATOM',
                 bond_pattern=b'@<TRIPOS>BOND'):

        self.mol_pattern = mol_pattern
        self.name_pattern = name_pattern
        self.atom_pattern = atom_pattern
        self.bond_pattern = bond_pattern
        self.count = 0

        super().__init__(fname=fname,
                         savedir=savedir,
                         atomtyper=atomtyper
                        )


    def update_patterns(self):
        if isinstance(self.file.peek(0), bytes):
            if isinstance(self.mol_pattern, str):
                self.mol_pattern = self.mol_pattern.encode()
            if isinstance(self.name_pattern, str):
                self.name_pattern = self.name_pattern.encode()
            if isinstance(self.atom_pattern, str):
                self.atom_pattern = self.atom_pattern.encode()
            if isinstance(self.bond_pattern, str):
                self.bond_pattern = self.bond_pattern.encode()
        else:
            if isinstance(self.mol_pattern, bytes):
                self.mol_pattern = self.mol_pattern.decode()
            if isinstance(self.name_pattern, bytes):
                self.name_pattern = self.name_pattern.decode()
            if isinstance(self.atom_pattern, bytes):
                self.atom_pattern = self.atom_pattern.decode()
            if isinstance(self.bond_pattern, bytes):
                self.bond_pattern = self.bond_pattern.decode()

    
    def open(self, fname=None):
        if self.check_open():
            self.close()
        if fname is not None:
            self.fname = fname
        if self.fname.endswith('.gz'):
            self.file = gzip.open(self.fname, 'rb')
        else:
            self.file = open(self.fname, 'r')
        self.endfile = False
        self.count = 0

        if self.savedir is None:
            self.savedir = os.path.join(os.path.dirname(self.fname), 'h5')

        self.update_patterns()
    
    def get_next_block(self):
        mtext = []
        molname = None
        while True:
            line = self.file.readline()
            self.endfile = (len(line) == 0)
            if self.endfile:
                break
            if line.startswith(self.mol_pattern):
                break
            if line.startswith(self.name_pattern):
                molname = line.strip()
            mtext.append(line)
        if isinstance(self.mol_pattern, bytes):
            mtext.insert(0,self.mol_pattern+b'\n')
        else:
            mtext.insert(0,self.mol_pattern+'\n')
        return mtext, molname

    
    def get_next_rdmol(self, sanitize=True, removeHs=True):
        if self.count == 0:
            header = self.get_next_block()
        mol, molname = self.get_next_block()            
        if isinstance(self.mol_pattern, bytes):
            mol = b''.join(mol).decode()
        else:
            mol = ''.join(mol)
        rdmol = Chem.rdmolfiles.MolFromMol2Block(mol, sanitize=sanitize, removeHs=False, cleanupSubstructures=False)
        if rdmol is None:
            print('failed to read molecule %d.\tTrying without sanitize...' %
                  (self.count))
            rdmol = Chem.MolFromMol2Block(mol, sanitize=sanitize)
        
        if removeHs:
            try:
                rdmol = Chem.RemoveHs(rdmol, updateExplicitCount=True, sanitize=False)
            except:
                print('mol\n',mol)
                raise Exception

        rdmol = Mol2(rdmol)
        if isinstance(molname, bytes):
            molname = molname.decode()
        rdmol.SetProp('zinc_id', molname)
        self.count += 1
        return rdmol


    def get_next_data_rdkit(self, canonicalize=True):
        moldata = {}

        rdmol2 = self.get_next_rdmol()
        moldata['names'] = rdmol2.get_zinc_id()
        moldata['coords'] = rdmol2.get_coords(canonical=canonicalize).astype(np.float32)
        moldata['atom_types'] = self.get_atom_types(rdmol2)
        
        bonds, bond_types = self.get_bonds_types(rdmol2)
        moldata['bonds'] = bonds
        moldata['bond_types'] = bond_types
        return moldata, rdmol2
 

    def skip_to_mol(self, molnum, fname=None):
        if fname is not None:
            self.fname = fname
            self.open()

        if self.check_open():
            if molnum < self.count:
                self.close()
                self.open()
        else:
            self.open()            

        while self.count < (molnum+1):
            pos = self.file.tell()
            line = self.file.readline()
            if not line:
                print('molnum is greater than file length!')
                return
            if line.startswith(self.mol_pattern):
                self.count += 1
        pos = pos + len(self.mol_pattern) + 1
        self.file.seek(pos)
        return pos


    def count_mols(self, fname=None, numstart=0):
        if fname is None:
            if not self.check_open():
                self.open()
        else:
            if isinstance(fname, str):
                self.open(fname)

        self.count = 0
        if numstart > 0:
            self.skip_to_mol(numstart)
        while True:
            line = self.file.readline()
            if not line:
                break
            if line.startswith(self.mol_pattern):
                self.count += 1
        self.file.close()
        return self.count - numstart


    def get_rdmols(self, fname=None, numstart=0, numread='all', sanitize=True, removeHs=True):
        if fname is None:
            if not self.check_open():
                self.open()
        else:
            if isinstance(fname, str):
                self.open(fname)

        self.data_dir = os.path.dirname(self.fname)

        readall = (numread == 'all')
        mols = []

        self.count = 0
        if numstart > 0:
            print('Starting at molecule: %d' % numstart)
            seekpos = self.skip_to_mol(molnum=numstart)
            self.count = numstart

        header = self.get_next_block()
        while True:
            rdmol = self.get_next_rdmol(sanitize=sanitize, removeHs=removeHs)
            mols.append(rdmol)
            if self.count % 1e3 == (1e3-1):
                print('%dk molecules processed' % (round(self.count/1e3, 0)))
            if self.endfile:
                break
            if not readall and (self.count-numstart) >= numread:
                break
        self.close()
        return mols





class BulkSDMolParser(MolParser):
    def __init__(self,
                 fname=None,
                 savedir=None,
                 atomtyper=None):
        '''Class for loading multi-molecule .sdf files into rdkit molecules.'''
        
        self.mol_from_mol_block_func = Chem.rdmolfiles.MolFromMolBlock
        
        self.count = 0
        super().__init__(fname=fname,
                         savedir=savedir,
                         atomtyper=atomtyper
                        )

    def open(self, fname=None):
        if fname is None:
            fname = self.fname

        self.endfile = False
        
        if fname.endswith('.gz'):
            with gzip.open(fname, 'rb') as fgz:
                tmp2 = fname.replace('.gz','')
                with open(tmp2, 'wb') as fout:
                    shutil.copyfileobj(fgz, fout)
            fname = tmp2
        self.mol_supplier = Chem.rdmolfiles.SDMolSupplier(fname)
        self.length = len(self.mol_supplier)
        self.fname = fname
        self.count = 0


    def close(self):
        self.length = None
        return


    def get_next_block(self):
        self.mtext = self.mol_supplier.GetItemText(self.count)
        self.molname = molname_from_molblock(self.mtext)
        self.count += 1
        if self.count == self.length:
            self.endfile = True
        return self.mtext, self.molname


    def mol_from_mol_block(self, molblock, sanitize=True, removeHs=True):
        rdmol = self.mol_from_mol_block_func(molblock, sanitize=sanitize, removeHs=False)
        if rdmol is None:
            print('failed to read molecule %d.\tTrying without sanitize...' %
                  (self.count))
            rdmol = self.mol_from_mol_block_func(molblock, sanitize=False)
        if removeHs:
            try:
                rdmol = Chem.RemoveHs(rdmol, updateExplicitCount=True, sanitize=False)
            except:
                print('mol\n',molblock)
                rdmol = None
                # raise Exception
        return rdmol


    def get_next_rdmol(self, sanitize=True, removeHs=True):
        if isinstance(self.mol_supplier, Chem.rdmolfiles.ForwardSDMolSupplier):
            rdmol = next(self.mol_supplier)
        else:
            molblock, molname = self.get_next_block()
            rdmol = self.mol_from_mol_block(molblock, sanitize=sanitize, removeHs=removeHs)

        if rdmol:
            rdmol = Mol2(rdmol)
            rdmol.SetProp('zinc_id', molname)
        return rdmol


    def get_next_data_rdkit(self, canonicalize=True):
        moldata = {}

        rdmol2 = self.get_next_rdmol()
        moldata['names'] = rdmol2.get_zinc_id()
        moldata['coords'] = rdmol2.get_coords(canonical=canonicalize).astype(np.float32)
        moldata['atom_types'] = self.get_atom_types(rdmol2)
        
        bonds, bond_types = self.get_bonds_types(rdmol2)
        moldata['bonds'] = bonds
        moldata['bond_types'] = bond_types
        return moldata, rdmol2
 

    def skip_to_mol(self, molnum, fname=None):
        if fname is not None:
            self.fname = fname
            self.open()

        if not self.check_open():
            self.open()            
        self.count = molnum


    def count_mols(self, fname=None, numstart=0):
        if fname is None:
            if not self.check_open():
                self.open()
        else:
            if isinstance(fname, str):
                self.open(fname)

        return len(self.mol_supplier) - numstart


    def get_rdmols(self, fname=None, numstart=0, numread='all', sanitize=True, removeHs=True):
        if fname is None:
            if not self.check_open():
                self.open()
        else:
            if isinstance(fname, str):
                self.open(fname)

        readall = (numread == 'all')
        mols = []

        self.count = 0
        if numstart > 0:
            print('Starting at molecule: %d' % numstart)
            self.skip_to_mol(molnum=numstart)

        while True:
            rdmol = self.get_next_rdmol(sanitize=sanitize, removeHs=removeHs)
            mols.append(rdmol)
            if self.count % 1e3 == (1e3-1):
                print('%dk molecules processed' % (round(self.count/1e3, 0)))
            if self.endfile:
                break
            if not readall and (self.count-numstart) >= numread:
                break
        self.close()
        return mols


class MolFilter(object):
    ATOMIC_NUMS_DEFAULT = ATOMIC_NUMS_DEFAULT
    FILTER_KEYS = ['natoms_min', 'natoms_max', 'width_min', 'width_max', 'weight_min', 'weight_max', 
                   'hbd_min', 'hbd_max', 'hba_min', 'hba_max', 'rotb_min', 'rotb_max', 'logp_min', 'logp_max',
                   'check_3d', 'atomic_nums', 'exclude_pains']
    def __init__(self, 
                 natoms_min : int=None, 
                 natoms_max : int=None,
                 width_min : float=None,
                 width_max : float=None,
                 weight_min : float=None,
                 weight_max : float=None,
                 hbd_min : int=None, 
                 hbd_max : int=None,
                 hba_min : int=None, 
                 hba_max : int=None,
                 rotb_min : int=None, 
                 rotb_max : int=None,
                 logp_min : float=None,
                 logp_max : float=None,                 
                 check_3d : bool=None,
                 atomic_nums: Iterable=None,
                 exclude_pains: bool = None,
                 ring_sizes: list[int] = None,          
                 ring_system_max : int=None,          
                 ring_loops_max : int=None,          
                 double_bond_pairs : bool=None,

                 ) -> None:
        '''Class for filtering sets of molecules based on their properties.'''
        
        self.natoms_min = natoms_min
        self.natoms_max = natoms_max

        self.width_min = width_min
        self.width_max = width_max

        self.weight_min = weight_min
        self.weight_max = weight_max

        self.logp_min = logp_min
        self.logp_max = logp_max
        
        self.hbd_min = hbd_min
        self.hbd_max = hbd_max
        
        self.hba_min = hba_min
        self.hba_max = hba_max
        
        self.rotb_min = rotb_min
        self.rotb_max = rotb_max

        self.check_3d = check_3d
        
        self.exclude_pains = exclude_pains
        
        self.ring_sizes = ring_sizes
        self.ring_system_max = ring_system_max
        self.ring_loops_max = ring_loops_max
        self.double_bond_pairs = double_bond_pairs

        if atomic_nums is not None:
            assert isinstance(atomic_nums, Iterable)
            atomic_nums = np.array(atomic_nums, dtype=int)

        self.atomic_nums = atomic_nums

    def __repr__(self):
        return self.__dict__.__repr__()

    @property
    def filter_params(self):
        return self.__dict__
    
    @property
    def filter_keys(self):
        return ['natoms_min', 'natom']

    @staticmethod
    def init_drug_like(exclude_pains=None, width_max=None, **kwargs):
        return MolFilter(weight_min = 200,
                      weight_max = 500,
                      natoms_min = 0,
                      natoms_max = 50,
                      hbd_max = 10,
                      hba_max = 5,
                      rotb_max = 12,  # rdkit tends to overestimate rotb
                      logp_min=-1., 
                      logp_max=5., 
                      atomic_nums=ATOMIC_NUMS_DEFAULT,
                      exclude_pains=exclude_pains,
                      width_max=width_max,
                      **kwargs
                     )
    
    @staticmethod
    def init_gen_opt(exclude_pains=None, width_max=None, **kwargs):
        return MolFilter(ring_sizes = [5,6],
                      ring_system_max = 3,
                      ring_loops_max = 0,
                      double_bond_pairs = False,
                      exclude_pains=exclude_pains,
                      width_max=width_max,
                      **kwargs
                     )

    @staticmethod
    def init_default(**kwargs):
        return MolFilter.init_drug_like(width_max=20, exclude_pains=False)


    def filter_mols(self, mols):
        '''Filter list of molecules, returning list of molecules that pass all filters'''
        if not isinstance(mols, list):
            mols = [mols]
        return [m for m in mols if self.check_mol(m)]



    def check_mols(self, mols):
        '''Check list of molecules, returning list of True/False for each molecule indicating if it passed all filters'''
        if not isinstance(mols, list):
            mols = [mols]
        return np.array([self.check_mol(m) for m in mols], dtype=bool)
    

    def check_mol(self, mol):
        assert isinstance(mol, Chem.Mol), 'mol must be of type rdkit.Chem.Mol'
        return all([self.filter_natoms(mol), 
                    self.filter_width(mol), 
                    self.filter_weight(mol), 
                    self.filter_logp(mol),
                    self.filter_hba(mol),
                    self.filter_hbd(mol),
                    self.filter_rotb(mol),
                    self.filter_3d(mol),
                    self.filter_atomic_nums(mol),
                    self.filter_pains(mol),
                    self.filter_rings(mol),
                    self.filter_bonds(mol),
                    ])
    
    @staticmethod
    def filter_val(mol, func, min_val, max_val):        
        is_valid = True
        if (min_val is not None) or (max_val is not None):
            mol_val = func(mol)
            if min_val is not None:
                is_valid *= (mol_val >= min_val)
            if max_val is not None:
                is_valid *= (mol_val <= max_val)
        return is_valid


    def filter_logp(self, mol):
        func = DEFAULT_PROPERTY_FUNCS['logp']
        return self.filter_val(mol, func, self.logp_min, self.logp_max)
    
    def filter_rotb(self, mol):
        func = DEFAULT_PROPERTY_FUNCS['rotb']
        return self.filter_val(mol, func, self.rotb_min, self.rotb_max)

    def filter_hba(self, mol):
        func = DEFAULT_PROPERTY_FUNCS['hba']
        return self.filter_val(mol, func, self.hba_min, self.hba_max)

    def filter_hbd(self, mol):
        func = DEFAULT_PROPERTY_FUNCS['hbd']
        return self.filter_val(mol, func, self.hbd_min, self.hbd_max)

    def filter_natoms(self, mol):
        func = DEFAULT_PROPERTY_FUNCS['natoms']
        return self.filter_val(mol, func, self.natoms_min, self.natoms_max)

    def filter_weight(self, mol):
        func = DEFAULT_PROPERTY_FUNCS['mw']
        return self.filter_val(mol, func, self.weight_min, self.weight_max)
    
    def filter_width(self, mol):
        func = DEFAULT_PROPERTY_FUNCS['width']
        return self.filter_val(mol, func, self.width_min, self.width_max)
    
    def filter_atomic_nums(self, mol):
        if self.atomic_nums is not None:
            for atom in mol.GetAtoms():
                anum = atom.GetAtomicNum()
                if anum != 1:
                    if anum not in self.atomic_nums:
                        return False
        return True


    def filter_3d(self, mol):
        if self.check_3d:
            return is_3d_mol(mol)
        return True

    def filter_pains(self, mol):
        if self.exclude_pains:
            if not hasattr(self, 'pains_patterns'):
                self.load_pains_patterns()

            for pattern in self.pains_patterns:
                if mol.HasSubstructMatch(pattern):
                    return False
        return True
    
    def filter_rings(self, mol):
        return self.check_rings(mol, self.ring_sizes, self.ring_system_max, self.ring_loops_max)
    
    def filter_bonds(self, mol):
        return self.check_bonds(mol, self.double_bond_pairs)
    
    
    def get_pains(self, mol):
        if not hasattr(self, 'pains_patterns'):
            self.load_pains_patterns()

        for i, pattern in enumerate(self.pains_patterns):
            if mol.HasSubstructMatch(pattern):
                return pattern, self.pains_ids[i]
        


    def load_pains_patterns(self, path='data/pains_filter/PAINS.sieve'):
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, 'PAINS.sieve'))
            path = files[0] if len(files) > 0 else path
        
        assert os.path.isfile(path), f'PAINS.sieve File Not Found at path: {path}'

        pains = []
        pids = []
        for line in open(path, 'r'):
            if line[0] == "#": continue # ignore header
            line = line.strip().split()
            if line:
                m = Chem.MolFromSmarts(line[2])
                pains.append(m)
                pids.append(line[1].replace('regId=',''))
        self.pains_patterns = pains
        self.pains_ids = pids

    def get_connected(self, ring_dict, i, root=None, visited=None, depth=0):
        '''DFS traversal to get connected components.'''
        if depth==0:
            root = i
        loop = False
        if visited is None:
            visited = []
        members = set()
        if (i in visited):
            loop = (depth > 2) and (i==root)
            return members, loop
        
        visited.append(i)
        for ri in ring_dict[i]:
            mem, lp = self.get_connected(ring_dict, ri, root, visited, depth=depth+1)
            members |= mem
            loop |= lp
            members.add(ri)
            visited.append(ri)
        if depth == 0:
            members = tuple(members)
        return members, loop


    def get_ring_systems(self, m, use_bonds=False, debug=False):
        '''Gets info about ring systems. Returns ring membership, size, and whether looped for each ring systems.'''
        if m is None:
            return [], [], []
        try:
            Chem.GetSSSR(m)
        except Exception as e:
            print(m)
            raise e
        if use_bonds:
            ring_inf = [x for x in m.GetRingInfo().BondRings()]
        else:
            ring_inf = [x for x in m.GetRingInfo().AtomRings()]

        # get adjacency as dict of node: set(neighbors)
        d = defaultdict(set)
        for i in range(len(ring_inf)):
            ring = ring_inf[i]
            for ri in ring:
                for k in range(len(ring_inf)):
                    if (k != i) and (ri in ring_inf[k]):
                        d[i].add(k)

        # graph traversal to get connected ring systems
        conn = {}
        for ri, rings in d.items():
            conn[ri] = self.get_connected(d, ri)
            
        ring_systems = list(set([x[0] for x in conn.values()]))  # tuples of ring numbers in each system
        loops = defaultdict(bool)
        for rsys, loop in conn.values():
            loops[rsys] |= loop

        # add info for singleton rings
        for i in range(len(ring_inf)):
            if i not in d.keys():
                ring_systems.append((i,))
                loops[(i,)] = False
        ring_sizes = {x:tuple([len(ring_inf[x2]) for x2 in x]) for x in ring_systems} # sizes of each ring in ring system
        
        
        if debug:
            print('d:', d)
            print('conn:', conn)
            print('loops:', loops)
            
        # return lists
        loops = [loops[x] for x in ring_systems]
        ring_sizes = [ring_sizes[x] for x in ring_systems]
        
        return ring_systems, ring_sizes, loops

    def df_get_ring_system_info(self, df):
        df['ring_loops'] = 0
        df['ring_system_max'] = 0
        df = df.reset_index(drop=True)
        for i in range(len(df)):
            m = df.rdmol.iloc[i]
            ring_systems, ring_sizes, loops = self.get_ring_systems(m)
            df.loc[i, 'ring_system_max'] = max([len(x) for x in ring_systems]+[0])
            df.loc[i, 'ring_loops'] = sum(loops)
        
        df.loc[:, 'ring_loops'] = df['ring_loops'].astype(int)
        df.loc[:, 'ring_system_max'] = df['ring_system_max'].astype(int)
        return df
            
        
    def check_rings(self, m, ring_sizes=None, ring_system_max=None, ring_loops_max=None):
        if ring_sizes is not None:
            Chem.GetSSSR(m)
            rs = [len(x) for x in m.GetRingInfo().AtomRings()]

            if (len(rs) > 0) and not all([x in ring_sizes for x in rs]):
                return False
        
        # check ring systems
        if ring_loops_max or ring_system_max:
            ring_systems, ring_sizes, loops = self.get_ring_systems(m)
            rsm = max([len(x) for x in ring_systems]+[0])
            num_loops = sum(loops+[False])
            if ring_system_max and (rsm > ring_system_max):
                return False
            if ring_loops_max and (num_loops > ring_loops_max):
                return False
            
        return True


    def check_bonds(self, m, double_bond_pairs=None):
        if double_bond_pairs is not None:
            # Check if mol has consecutive double bonds.
            for atom in m.GetAtoms():
                num_db = sum([b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()])
                if num_db > 1:
                    return False
        return True


class MolDatasetGenerator(object):
    def __init__(self, molparser, extension=None, molfilter=None) -> None:       

        self.extension = extension
        self.molparser = molparser
        self.molfilter = molfilter
        pass

    def save_coords_types(self,
                          file,
                          savedir,
                          savename=None,
                          numstart=0,
                          chunks=1000,
                          numread='all',
                          proportion_read=1.,
                          mol_natoms_max=80,
                          center_mols=True,
                          compression=None,
                          verbose=True,
                          debug=False,
                          **kwargs
                          ):        
        '''Class for generating h5py dataset.'''

        self.molparser.fname = file
        self.molparser.savedir = savedir
        os.makedirs(self.molparser.savedir, exist_ok=True)
        readall = (numread == 'all')

        if savename is None:
            savename = file
        savename = os.path.basename(savename)
        
        mol_bonds_max = int(2*mol_natoms_max)
        types_bonds_max = 10

        types_atoms_max = 10
        
        

        nmols = self.molparser.count_mols(numstart=numstart)

        if numread != 'all':
            nmols = min(nmols, numread)

        nmols_read = int(nmols * proportion_read)
        read_every = nmols // nmols_read

        
        chunks = min(chunks, nmols_read)

        print('nmols', nmols_read)
        self.molparser.open()

        fname = os.path.basename(file)
        savename = savename[:savename.rfind(self.extension)] + 'h5'
        print('Saving hdf5 data to: {savename}')
        actual_natoms_max = 0
        actual_atom_types_max = 0
        with h5py.File(os.path.join(savedir, savename), 'w') as f5:
            dat_atom_types = f5.create_dataset("atom_types", 
                                          shape=(nmols_read, mol_natoms_max, types_atoms_max), 
                                          chunks=(chunks, mol_natoms_max, types_atoms_max), 
                                          compression=compression,
                                          dtype='int16',
                                          fillvalue=-1,
                                          **kwargs
                                          )

            dat_coords = f5.create_dataset("atom_coords", 
                                           shape=(nmols_read, mol_natoms_max, 3), 
                                           chunks=(chunks, mol_natoms_max, 3), 
                                           compression=compression,
                                           dtype='float32', 
                                           fillvalue=np.nan,
                                          **kwargs
                                          )
                       
            dt = h5py.special_dtype(vlen=str)
            dat_names = f5.create_dataset("names", 
                                           shape=(nmols_read), 
                                           chunks=(chunks), 
                                           compression=compression,
                                           dtype=dt,
                                          **kwargs
                                         )


            if numstart > 0:
                print('Starting at molecule: %d' % numstart)
                _ = self.molparser.skip_to_mol(molnum=numstart)
            
            i = 0
            mol_n = 0
            n_mols_processed = 0
            n_mols_valid = 0
            while not self.molparser.endfile:
                if read_every > 1:
                    mol_n = int(read_every * i)
                    if mol_n >= nmols:
                        break
                    _ = self.molparser.skip_to_mol(molnum=mol_n)

                try:
                    moldata, rdmol = self.molparser.get_next_data_rdkit(canonicalize=center_mols)
                except:
                    print('i',i, '   count:',self.molparser.count)
                    raise Exception
                
                n_mols_processed += 1
                if self.molfilter is not None:
                    if not self.molfilter.check_mol(rdmol):
                        continue
                
                n_mols_valid += 1
                n_atoms = len(moldata['atom_types'])
                
                dat_names[i] = moldata['names']
                dat_coords[i, :n_atoms] = moldata['coords']
                
                actual_natoms_max = max(actual_natoms_max, n_atoms)
                
                for ia in range(len(moldata['atom_types'])):
                    t = moldata['atom_types'][ia]
                    dat_atom_types[i, ia, :len(t)] = t
                    actual_atom_types_max = max(actual_atom_types_max, len(t))
                
                i += 1
                if debug:
                    print('mol:', self.molparser.count)
                    
                if verbose and ((i+1) % max(chunks,1000) == 0):
                    print('molecules %dk processed' % (round(self.molparser.count/1e3, 0)))

                if not readall:
                    if i >= numread:
                        break
            
            if i < nmols_read:
                dat_coords = dat_coords[:i]
                dat_names = dat_names[:i]
                dat_atom_types = dat_atom_types[:i]
        self.molparser.close()
        return (n_mols_processed, n_mols_valid)


class BaseAtomTyper(object):
    def __init__(self, ntypes : int = 1024):
        """Atom typer class. Defines a mapping of atom/bond types to a 1hot vector.
        """    
        self.ntypes = ntypes
        self.t2n = {}
        self.n2t = {}
        self.groups = {}
        pass

    def __len__(self):
        return self.ntypes

    @property
    def group_names(self):
        return list(self.groups.keys())

    def type2num(self, value):
        try:
            # backward compatibility with elements as ints instead of 'e[#]'
            v = int(value)
            if v >= 1 and v <= 118:
                value = 'e'+str(v)
            else:
                raise Exception
        except:
            pass

        return self.t2n[value]

    def num2type(self, value):
        return self.n2t[int(value)]

    def get_group(self, group):
        return self.groups[group]

    def check_init_ptable(self):
        '''initializes rdkit periodic table here to avoid pickling issue in pytorch dataloader'''
        if not hasattr(self, 'ptable'):
            self.ptable = Chem.rdchem.GetPeriodicTable()

    def element_symbol2num(self, symbol):
        self.check_init_ptable()
        return self.ptable.GetAtomicNumber(symbol)

    def element_num2symbol(self, num):
        self.check_init_ptable()
        return self.ptable.GetElementSymbol(int(num))

    def types2vec(self, types):
        '''
        converts a list of types to a 1hot types vector
        '''
        if isinstance(types, torch.Tensor):
            types = types.tolist()

        if isinstance(types[0], str):
            tvec = torch.zeros(self.ntypes, dtype=bool)
            for t in types:
                tnum = self.type2num(t)
                tvec[tnum] = 1
        elif isinstance(types[0][0], str):
            tvec = torch.zeros(len(types), self.ntypes, dtype=bool)
            for i in range(len(types)):
                for t in types[i]:
                    tnum = self.type2num(t)
                    tvec[i, tnum] = 1
        elif isinstance(types[0][0][0], str):
            tvec = torch.zeros(len(types), len(types[0]), self.ntypes, dtype=bool)
            for i in range(len(types)):
                for j in range(len(types[i])):
                    for t in types[i][j]:
                        tnum = self.type2num(t)
                        tvec[i,j,tnum] = 1

        return tvec

    def nums2vec(self, types):
        '''
        converts a vector of type nums to a 1hot types vector
        '''
        if isinstance(types, torch.Tensor):
            types = types.int().tolist()
        if isinstance(types, np.ndarray):
            types = types.astype(int).tolist()

        try:
            numeric_types = (int, np.int16, np.int32, np.int64)
            if isinstance(types[0], numeric_types):
                tvec = torch.zeros(self.ntypes, dtype=bool)
                for tnum in types:
                    if tnum > 0:
                        tvec[tnum] = 1
            elif isinstance(types[0][0], numeric_types):
                tvec = torch.zeros(len(types), self.ntypes, dtype=bool)
                for i in range(len(types)):
                    for tnum in types[i]:
                        if tnum > 0:
                            tvec[i, tnum] = 1
            elif isinstance(types[0][0][0], numeric_types):
                tvec = torch.zeros(len(types), len(types[0]), self.ntypes, dtype=bool)
                for i in range(len(types)):
                    for j in range(len(types[i])):
                        for tnum in types[i][j]:
                            if tnum > 0:
                                tvec[i,j,tnum] = 1
        except Exception as e:
            print('\nERROR:')
            print('\ntypes:', types)
            print([type(x) for x in types])
            raise e
        return tvec

    def _vec2nums(self, vec):
        assert vec.ndim ==1, 'input vec must be 1-dimensional' 
        return torch.arange(self.ntypes)[vec == 1].numpy().tolist()

    def vec2nums(self, vec):
        '''
        converts a 1hot encoded types vector to an array of type numbers
        '''
        b = []
        if vec.ndim == 1:
            b = self._vec2nums(vec)
        elif vec.ndim == 2:
            for i in range(len(vec)):
                b.append(self._vec2nums(vec[i]))
        elif vec.ndim == 3:
            for i in range(len(vec)):
                b.append([self._vec2nums(v) for v in vec[i]])
        return b


    def vec2types(self, vec):
        '''
        converts a 1hot encoded types vector to a list of types
        '''
        b = []
        tnums = self.vec2nums(vec)
        if isinstance(tnums[0], (int, np.int16, np.int32)):
            for tnum in tnums:
                b.append(self.num2type(tnum))
        elif isinstance(tnums[0][0], (int, np.int16, np.int32)):
            t1 = []
            for i, tlist in enumerate(tnums):
                t1 = [self.num2type(tnum) for tnum in tlist]
                b.append(t1)
        elif isinstance(tnums[0][0][0], (int, np.int16, np.int32)):
            for tlist1 in tnums:
                t1 = []
                for tlist2 in tlist1:
                    t2 = [self.num2type(tnum) for tnum in tlist2]
                    t1.append(t2)
                b.append(t1)

        return b
    
    def update_luts(self, type2num_dict: dict) -> None:
        self.t2n = type2num_dict
        self.n2t = {v: k for k, v in self.t2n.items()}

    def update_groups(self) -> None:
        groups = {}
        groups['elements_all'] = self.elements_dict
        groups['atom_features_all'] = self.atom_features_dict
        groups['bonds_all'] = self.bond_dict
        groups['bond_features_all'] = self.bond_features_dict
        groups['bto_all'] = self.bto_dict

        self.groups = groups
        for g, vals in groups.items():
            setattr(self, g, vals)

    def bond_order2type(self, order):
        bond_order_lut = {1:'bond1',
                    2:'bond2',
                    3:'bond3',
                    4:'bond4',
                    1.5:'bondar'}
        
        return bond_order_lut[order]

    def is_element(self, key : Union[int,str]):
        '''
        Check if `key` is an atomic element descriptor.

        key : int or str 
             If `key` is a str, checks if it is in an element descriptor name.
             If `key` is an int, checks if it is an element descriptor number.
        '''
        if isinstance(key, str):
            return key in self.elements_dict.keys()
        else:
            return int(key) in self.elements_dict.values()

    def is_bond(self, key : Union[int,str]):
        '''
        Check if `key` is a bond descriptor.

        key : int or str 
             If `key` is a str, checks if it is in an bond descriptor name.
             If `key` is an int, checks if it is an bond descriptor number.
        '''
        if isinstance(key, str):
            return key in self.bond_dict.keys()
        else:
            return int(key) in self.bond_dict.values()

    def is_atom_feature(self, key : Union[int,str]):
        '''
        Check if `key` is a atom feature descriptor.

        key : int or str 
             If `key` is a str, checks if it is in an atom feature descriptor name.
             If `key` is an int, checks if it is an atom feature descriptor number.
        '''
        if isinstance(key, str):
            return key in self.atom_features_dict.keys()
        else:
            return int(key) in self.atom_features_dict.values()


    def is_bond_feature(self, key : Union[int,str]):
        '''
        Check if `key` is a bond feature descriptor.

        key : int or str 
             If `key` is a str, checks if it is in an bond feature descriptor name.
             If `key` is an int, checks if it is an bond feature descriptor number.
        '''
        if isinstance(key, str):
            return key in self.bond_features_dict.keys()
        else:
            return int(key) in self.bond_features_dict.values()


class AtomTyperCustom(BaseAtomTyper):
    def __init__(self, 
                 ntypes: int,
                 elements_dict: dict = {},
                 atom_features_dict: dict = {},
                 bonds_dict: dict = {},
                 bond_features_dict: dict = {},
                 bto_dict: dict = {},
                 **other_types: dict,
                 ) -> None:
        """Custom atom typer class. Can be used to specify a custom mapping of atom/bond types to a 1hot vector.

        Args:
            ntypes (int): length of 1hot type vector.  Defaults to 1024.
            elements_dict (dict, optional): Atomic number (element) types. Defaults to {}.
            atom_features_dict (dict, optional): Atom feature types. Defaults to {}.
            bonds_dict (dict, optional): Bond types. Defaults to {}.
            bond_features_dict (dict, optional): Bond feature types. Defaults to {}.
            bto_dict (dict, optional): Bonded to types. Defaults to {}.
        """            
        super().__init__(ntypes)

        self.elements_dict = elements_dict
        self.atom_features_dict = atom_features_dict
        self.bto_dict = bto_dict
        self.bonds_dict = bonds_dict
        self.bond_features_dict = bond_features_dict

        d = {}
        d.update(self.elements_dict)
        d.update(self.bto_dict)
        d.update(self.bonds_dict)
        d.update(self.atom_features_dict)
        d.update(self.bond_features_dict)

        for key,vals in other_types.items():
            setattr(self, key, vals)
            d.update(getattr(self,key))
            self.groups[key.replace('_dict','')] = vals

        self.update_luts(d)
        self.update_groups()


class AtomTyperDefault(BaseAtomTyper):
    def __init__(self) -> None:
        """Default atom typer class. Defines a default mapping of atom/bond types to a 1hot vector.
        """        
        super().__init__()
        self.init_type_specs()
        

    def init_type_specs(self) -> None:
        d = {}

        d['default'] = 1023
        d['unknown_atom'] = 1023

        ### atomic numbers (elements)
        i0 = 0
        elements_dict = {}
        for i in range(1, 119):
            elements_dict['e'+str(i)] = i + i0

        d.update(elements_dict)
        self.elements_dict = elements_dict

        ### bonded to
        i0 = 119
        bto_dict = {}
        for i in range(0, 118):
            bto_dict[f'bto{i+1}'] = i+i0
        d.update(bto_dict)
        self.bto_dict = bto_dict


        ### atom_features
        i0 = 301
        atom_features = ['HBD', 'HBA', 'Aromatic', 'FChargePos', 'FChargeNeut', 'FChargeNeg']
        atom_features_dict = {}

        for i in range(len(atom_features)):
            atom_features_dict[atom_features[i]] = i0 + i
        d.update(atom_features_dict)
        self.atom_features_dict = atom_features_dict


        ### bonds
        i0 = 800
        d['unknown_bond'] = i0
        bond_dict = {'bond1': i0+1,  # single bond
                     'bond2': i0+2,  # double bond
                     'bond3': i0+3,  # triple bond
                     'bond4': i0+4,  # quadruple bond (rare)
                     'bondar': i0+5,  # aromatic bond (order=1.5)
                     'bondh': i0+6,  # hydrogen bond
                     'bondi': i0+7,  # ionic bond
                     }
        d.update(bond_dict)
        self.bond_dict = bond_dict


        ### bond_features
        i0 = 820
        bond_features = ['bondring']
        bond_features_dict = {}

        for i in range(len(bond_features)):
            bond_features_dict[bond_features[i]] = i0 + i
        d.update(bond_features_dict)
        self.bond_features_dict = bond_features_dict

        self.update_luts(d)
        self.update_groups()