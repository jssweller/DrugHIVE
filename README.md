# DrugHIVE: Structure-based drug design with a deep hierarchical generative model

[![DOI](https://img.shields.io/badge/DOI-BioRxiv-B31B1B.svg)](https://www.biorxiv.org/content/10.1101/2023.12.22.573155v2)

This repository is the official implementation of **DrugHIVE**, a deep hierarchical variational autoencoder developed for structure-based drug design. [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.12.22.573155v2).


 <div>
  <a href="https://github.com/jssweller/drughive#gh-light-mode-only">
         <img width="100%" style="aspect-ratio: auto; max-width: 1000px" src="https://github.com/jssweller/DrugHIVE/blob/main/img/hive.png#gh-light-mode-only"/>
  </a>

  <a href="https://github.com/jssweller/drughive#gh-dark-mode-only">
         <img width="100%" style="aspect-ratio: auto; max-width: 1000px" src="https://github.com/jssweller/DrugHIVE/blob/main/img/hive_dark.png#gh-dark-mode-only"/>
  </a>
 </div>


# Installation

### 1. Dependencies

The code has been tested in the following environment:
| Software          | Version   |
|-------------------|-----------|
| Python            | 3.9.16    |
| CUDA              | 11.6      |
| OpenBabel         | 3.1.1     |
| PyTorch           | 1.12.1    |
| PyTorch Lightning | 2.0.0     |
| RDKit             | 2021.09.5 |

### Install via conda

Install dependencies using the listed requirements in `requirements.txt`:

```bash
conda create -n drughive -c conda-forge -c pytorch -c nvidia -c rdkit --file requirements.txt
```

### 2. Git clone the repository
```bash
git clone https://github.com/jssweller/DrugHIVE
```

# Sampling

### Pre-trained models

Pre-trained model weights can be downloaded from Zenodo:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12668687.svg)](https://doi.org/10.5281/zenodo.12668687)

```bash
wget -P model_checkpoints/ https://zenodo.org/records/12668687/files/drughive_model_ch9.ckpt
```


## Ligand generation
To sample from DrugHIVE, first adjust the parameters in the `generate.yml` example configuration file. Then, run the following command:

```bash
python generate_molecules.py config/generate.yml
```

To sample from the prior, set `zbetas: 1.` in the configuration file.  
To sample from the posterior, set `zbetas: 0.` in the configuration file.  
To sample between the prior and posterior, set the values of `zbetas` between `0.` and `1.`.


## Substructure modification (scaffold hopping)

To generate molecules with substructure modification, first adjust the parameters in the `generate_spatial.yml` example configuration file. Then, run the following command:

```bash
python generate.py config/generate_spatial.yml
```


## Ligand optimization


### Install QuickVina 2

Before running the optimization process, the QuickVina 2 docking tool must be installed:
  - download (or compile) the QuickVina2 docking tool from [https://qvina.github.io]( https://qvina.github.io)
  - place `qvina2.1` in `DrugHIVE/` or in a directory in listed in your `PATH` variable (e.g., `/usr/bin/`)

### Run optimization
To optimize molecules with DrugHIVE, first adjust the parameters in the `generate_optimize.yml` example configuration file. Then, run the following command:

```bash
python generate_optimize.py config/generate_optimize.yml
```


# Training

### 1. Download PDBbind dataset

Download and extract the PDBbind refined dataset from [http://www.pdbbind.org.cn/](http://www.pdbbind.org.cn/)

### 2. Download ZINC molecules

Download ZINC molecules from [https://zinc20.docking.org/](https://zinc20.docking.org/) in SDF or MOL2 format. Place them in a single directory.

### 3. Process the datasets for training

To process the PDBbind dataset, run:

```bash
python process_pdbbind_data.py <path/to/PDBbind/directory>
```

To process the ZINC dataset, run:

```bash
python process_zinc_data.py <path/to/ZINC/directory> -o data/zinc_data/zinc_data.h5 -ext <file_extension>
```
Here, `<file_extension>` can be one of `sdf` or `mol2`.

### 4. Run training

First, adjust the training parameters in the `config/train.yml` example configuration file. Make sure to set `data_path_pdb` and `data_path_zinc` to the locations of your datasets. Then, run the following command:

```bash
python train.py config/train.yml
```
