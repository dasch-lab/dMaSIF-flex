import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import *
from pathlib import Path
from Bio import SeqIO
import os
import re
import pandas as pd 
from Bio import pairwise2
from collections import defaultdict
#from ImmuneBuilder import ABodyBuilder2
#from ImmuneBuilder.util import sequence_dict_from_fasta
#from ImmuneBuilder.constants import restypes

try:
    from src.data_preprocessing.ESMFold import ESMModel, parse_pdb_b_factors_mean, parse_pdb_b_factors_binary, parse_pdb_b_factors_atomic
    from src.data_preprocessing.PDB import change_res_id
    from src.data_preprocessing.atomic_dictionary import hydrogen_to_heavy
except Exception:
    from ESMFold import ESMModel, parse_pdb_b_factors_mean, parse_pdb_b_factors_binary, parse_pdb_b_factors_atomic
    from PDB import change_res_id
    from atomic_dictionary import hydrogen_to_heavy

path_esm = "./src/provaESMFold"
parser = PDBParser(QUIET=True)

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}
count = 5

def match_df(csv_files, target_pdb_id, prefix):
    matching_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        matches = df[df['pdb'] == prefix + target_pdb_id+'.pdb']
        matching_dfs.append(matches)

    # Concatenate all matching rows into one DataFrame
    result_df = pd.concat(matching_dfs, ignore_index=True)
    return result_df

def computeFlexibility(fname, binary = False, atomic = False, itsflexible = False):
    fname = str(fname)
    model = ESMModel()
    pdb_name = fname.split('/')[-1].split('_')[0]
    b_factor_dict = {}
    sequence_dict = {}
    

    for record in SeqIO.parse(fname, "pdb-atom"):
        chain = record.id[-1]
        print(chain)
        sequence_dict[chain] = str(record.seq)
        print(len(str(record.seq)))
    
    for chain in sequence_dict.keys():
        out_file = pdb_name.split('.')[0] + '_' + chain + '_ESMFold.pdb'
        path = '/'.join([path_esm, out_file])
        if os.path.exists(path):
            pass
        else:
            output = model.generate_model(chain = chain, data=sequence_dict[chain], pdb_write=True, model_path=path)
            change_res_id(path)

        structure = parser.get_structure(path.split('.')[0], path)
        if binary:
            b_factor_dict_new = parse_pdb_b_factors_binary(structure)
        elif atomic:
            b_factor_dict_new = parse_pdb_b_factors_atomic(structure)
        else:
            b_factor_dict_new = parse_pdb_b_factors_mean(structure)
        b_factor_dict.update(b_factor_dict_new)
        
        print(len(b_factor_dict))
        
    
    return b_factor_dict



def load_structure_np(fname, center, binary = False, atomic = False):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    print(f'Processing {fname}')
    change_res_id(fname)
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()
    
    b_factor_dict = computeFlexibility(fname, binary, atomic)

    coords = []
    types = []
    plddt = []
    count = 0
    #plddt = 0
    fine = True
    for atom in atoms:
        coords.append(atom.get_coord())
        try:
            types.append(ele2num[atom.element])
        except KeyError:
            count = count +1
            ele2num[atom.element] = count
            types.append(ele2num[atom.element])
            print(ele2num)
        if atomic: 
            key = (atom.get_parent().get_parent().id, atom.get_parent().get_id()[1], atom.get_parent().get_resname(), atom.get_id())
        else:
            key = (atom.get_parent().get_parent().id, atom.get_parent().get_id()[1], atom.get_parent().get_resname())
        try:
            plddt.append(b_factor_dict[key])
        except KeyError:
            #if not atomic and not binary:
                #print('Error, I had to put 0')
                #plddt.append(0)
            if atomic:
                if atom.element == 'H':
                    heavy_atom_key = None
                    atom_name = atom.get_id()
                    resname = atom.get_parent().get_resname() 

                    
                    try:
                        heavy_atoms = hydrogen_to_heavy[resname]
                        if atom_name.startswith('H1') or atom_name.startswith('H2') or atom_name.startswith('H3'):
                            heavy_atom_key = (atom.get_parent().get_parent().id, 
                                            atom.get_parent().get_id()[1], 
                                            atom.get_parent().get_resname(), 
                                            'N')  # N-terminal nitrogen
                            plddt.append(b_factor_dict[heavy_atom_key])
                        elif atom_name in heavy_atoms:
                            heavy_atom_name = heavy_atoms[atom_name]
                            heavy_atom_key = (
                                atom.get_parent().get_parent().id,  # Chain ID
                                atom.get_parent().get_id()[1],       # Residue number
                                resname,                             # Residue name
                                heavy_atom_name                      # Heavy atom name
                            )
                            plddt.append(b_factor_dict[heavy_atom_key])
                        else:
                            print('Atom not found')
                            fine = False
                            print(f'Issues in {fname} for {key}')
                            continue
                        
                    except KeyError:
                            print(f"No heavy atom mapping found for {resname} {atom_name}")
                            fine = False
                            continue
                else:
                    print(f"Not an hidrogen")
                    plddt.append(1.0)
                    continue
            else:
                print(f"Not atomic search")
                fine = False
                print(f'Issues in {fname} for {key}')
                continue

                        

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)
    
    if fine:
        return {"xyz": coords, "types": types_array, "flexibility": plddt}
    else:
        return {"xyz": coords, "types": types_array, "flexibility": None}

def align_and_get_indices(cdr_seq, full_seq):
    alignments = pairwise2.align.localms(full_seq, cdr_seq, 2, -1, -0.5, -0.1)
    if not alignments:
        return pd.NA, pd.NA
    best_alignment = alignments[0]  # highest scoring alignment
    start = best_alignment.start + 1
    end = best_alignment.end  # inclusive
    return start, end

def get_label_for_residue(residue_index, dict_pos_label):
    for (start, end), label in dict_pos_label.items():
        if start <= residue_index <= end:
            return label
    return 0  # or None, if outside any CDR
    

def extract_cdrs_residue(df, fname, chain_name):
    from Bio import SeqIO

    dict_pos_label = defaultdict(dict)
    sequence_dict = {}

    # Parse PDB sequence
    for record in SeqIO.parse(fname, "pdb-atom"):
        chain = record.id[-1]
        sequence_dict[chain] = str(record.seq)


    # Align and store CDR regions + label
    for _, el in df.iterrows():
        try:
            start, end = align_and_get_indices(el['cdr3'], sequence_dict[el['chain']])
            label = el['preds']
            dict_pos_label[el['chain']][(start, end)] = label
        except Exception as e:
            print(f'Error: {fname} | {e}')
            continue

    return dict_pos_label
        
    
    
def load_structure_np_itsflexible(fname, name, chain, center, csv_files, prefix):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    print(f'Processing {fname}')
    #change_res_id(fname)
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()
    
    df_chains = match_df(csv_files, name+'_'+chain, prefix)
    
    if len(df_chains) == 0:
        all_zero = True
    else:
        all_zero = (df_chains['preds'] == 0).all()
    
    if not all_zero:
        dict_pos_label =  extract_cdrs_residue(df_chains, fname, chain)

    coords = []
    types = []
    plddt = []
    count = 0
    fine = True
      
    for atom in atoms:
        coords.append(atom.get_coord())
        try:
            types.append(ele2num[atom.element])
        except KeyError:
            count = count +1
            ele2num[atom.element] = count
            types.append(ele2num[atom.element])
            print(ele2num)
            
        if all_zero:
            #plddt.append(0)
            plddt.append(0.34)
        else:
            try:
                dict_chain = dict_pos_label[atom.get_parent().get_parent().id]
                label = get_label_for_residue(atom.get_parent().get_id()[1], dict_chain)
                label_new = 0.34 if label <= 0.34 else 1
                #plddt.append(label)
                plddt.append(label_new)
            except Exception:
                print(f'Error for {atom.get_parent().get_id()[1]}')
                #plddt.append(0)
                plddt.append(0.34)
                continue               

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)
    
    if fine:
        return {"xyz": coords, "types": types_array, "flexibility": plddt}
    else:
        return {"xyz": coords, "types": types_array, "flexibility": None}


def convert_pdbs(pdb_dir, npy_dir, list_names = None, binary = False, atomic = False):
    print("Converting PDBs")
    os.makedirs(npy_dir, exist_ok=True)
    pdb_files = list(pdb_dir.glob("*_*.pdb"))
    #pdb_names = []

    """if list_names is not None:
        with open(list_names, 'r') as f:
            pdb_names = [line.strip().split('_')[0] for line in f if line.strip()]  # Read non-empty lines"""

    for p in tqdm(pdb_files, desc="Processing PDB files", total=len(pdb_files)):

        if "_" not in p.stem:
            print(f'There is no _ in the name of {p.name}')
            continue

        if binary:
            output_flex_file = npy_dir / f"{p.stem}_atomflex_bin.npy"
        elif atomic:
            output_flex_file = npy_dir / f"{p.stem}_atomflex_atom.npy"
        else:
            output_flex_file = npy_dir / f"{p.stem}_atomflex.npy"
        
        # Check if the output file already exists
        """if output_flex_file.exists():
            print(f"File {output_flex_file} already exists, skipping {p.name}")
            continue  # Skip to the next .pdb file"""

        if list_names and p.name.split('_')[0] not in list_names:
            continue

        try:
            protein = load_structure_np(p, center=False, binary = binary, atomic = atomic)
            np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
            np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])
            if protein["flexibility"] is not None:
                if binary:
                    np.save(npy_dir / (p.stem + "_atomflex_bin.npy"), protein["flexibility"])
                elif atomic:
                    np.save(npy_dir / (p.stem + "_atomflex_atom.npy"), protein["flexibility"])
                else:
                    np.save(npy_dir / (p.stem + "_atomflex.npy"), protein["flexibility"])
        except:
            print(f'Issue with load_sequence_np')

def convert_pdbs_Ab(pdb_dir, npy_dir, list_names = None, binary = False, atomic = False, itsflexible = False, csv_files=None):
    print("Converting PDBs")
    os.makedirs(npy_dir, exist_ok=True)
    pdb_files = list(pdb_dir.glob("*_*.pdb"))
    #pdb_names = []

    """if list_names is not None:
        with open(list_names, 'r') as f:
            pdb_names = [line.strip() for line in f if line.strip()]  # Read non-empty lines"""
    
    list_names_new = []
    
    for el in list_names:
        name = el.split('_')[0]
        chain1 = el.split('_')[1]
        chain2 = el.split('_')[2]
        list_names_new.append(name + '_' + chain1 + '.pdb')
        list_names_new.append(name + '_' + chain2 + '.pdb' )
    
    #list_names_new = list_names_new[5140:]
    check_null = 0
    for p in tqdm(list_names_new, desc="Processing PDB files", total=len(list_names_new)):
        
        
        pdb_path = pdb_dir / p  # Create full path for pdb file
        name, chain = p.split('.')[0].split('_')

        # Convert the string filename to a Path object for attribute access
        p = Path(pdb_path)
        
        if itsflexible:
            output_flex_file = npy_dir / f"{p.stem}_itsflexible.npy"
        else:
            if binary:
                output_flex_file = npy_dir / f"{p.stem}_atomflex_bin.npy"
            elif atomic:
                output_flex_file = npy_dir / f"{p.stem}_atomflex_atom.npy"
            else:
                output_flex_file = npy_dir / f"{p.stem}_atomflex.npy"
        
        """# Check if the output file already exists
        if output_flex_file.exists():
            print(f"File {output_flex_file} already exists, skipping {p.name}")
            continue  # Skip to the next .pdb file"""

        try:
            if itsflexible:
                protein = load_structure_np_itsflexible(p, name, chain, center=False, csv_files=csv_files, prefix = pdb_dir)
            else:
                protein = load_structure_np(p, center=False, binary = binary, atomic = atomic, itsflexible = itsflexible)
            
            #np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
            #np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])
                
            if protein["flexibility"] is not None:
                if itsflexible:
                    np.save(npy_dir / (p.stem + "_itsflexible.npy"), protein["flexibility"])
                else:
                    if binary:
                        np.save(npy_dir / (p.stem + "_atomflex_bin.npy"), protein["flexibility"])
                    elif atomic:
                        np.save(npy_dir / (p.stem + "_atomflex_atom.npy"), protein["flexibility"])
                    else:
                        np.save(npy_dir / (p.stem + "_atomflex.npy"), protein["flexibility"])
        except:
            print(f'Issue with load_sequence_np')