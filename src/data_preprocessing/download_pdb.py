import Bio
from Bio.PDB import * 
from Bio.SeqUtils import IUPACData
import sys
import importlib
import os
import numpy as np
from subprocess import Popen, PIPE
from pathlib import Path
from src.data_preprocessing.convert_pdb2npy import load_structure_np
import argparse
from tqdm import tqdm



def export_lib(new_path, name):
    current_path = os.environ.get("PATH", "")
    if current_path:
        new_path = current_path + ":" + new_path
    os.environ["PATH"] = new_path
    print('Added path ' + name)

def delete_folder(folder_path):
    try:
        # Remove the directory
        os.rmdir(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except OSError as e:
            print(f"Error: {folder_path} : {e.strerror}")

export_lib("/disk1/fingerprint/librerie/reduce/build/reduce/reduce_src/", 'reduce')
os.environ["REDUCE_HET_DICT"] = "/disk1/fingerprint/librerie/reduce/reduce_wwPDB_het_dict.txt"
folder_path = "/disk1/fingerprint/provaESMFold"

# Check if the folder already exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument(
    "--pdb", type=str,default='', help="PDB code along with chains to extract, example 1ABC_A_B", required=False
)
parser.add_argument(
    "--pdb_list", type=str,default='/disk1/fingerprint/SAbDab_preparation/sabdab_summary_output_filtered.txt', help="Path to a text file that includes a list of PDB codes along with chains, example 1ABC_A_B", required=False
)

tmp_dir = Path('./tmp')
pdb_dir = Path('/disk1/fingerprint/SAbDab_preparation/all_structures/raw')
npy_dir = Path('/disk1/fingerprint/SAbDab_preparation/all_structures/npys')
out_dir = Path('/disk1/fingerprint/SAbDab_preparation/all_structures/protonate')

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1" 


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extractPDB(
    infilename, outfilename, chain_ids=None 
):
    # extract the chain_ids from infilename and save in outfilename. 
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()

def remove_files_raw(folder_path, pdb_id):
    # List all files in the folder
    files = os.listdir(folder_path)
    # Iterate through the files
    for file in files:
        # Check if the file starts with the PDB ID
        if file.startswith(pdb_id):
            # Construct the full file path
            file_path = os.path.join(folder_path, file)
            # Remove the file
            os.remove(file_path)



def get_single(pdb_id: str, pdb_prot: str, chains: list):
    pdb_filename = pdb_dir/f"{pdb_id}.pdb"
    protonated_file = out_dir/f"{pdb_id}.pdb"
    pattern = f"{pdb_id}_"
    final_dir = os.listdir(npy_dir)

    file_exists = any(filename.startswith(pattern) for filename in final_dir)

    if not file_exists:
        print(f"dMaSIF features for {pdb_id} do not exists")

        #if not protonated_file.exists():
        #    # Download pdb 
        #    pdbl = PDBList()
        #    pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=tmp_dir,file_format='pdb')

            ##### Protonate with reduce, if hydrogens included.
            # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
        
        protonate(pdb_filename, protonated_file)

        pdb_filename = protonated_file

        # Extract chains of interest.
        for chain in chains:
            out_filename = pdb_dir/f"{pdb_id}_{chain}.pdb"
            extractPDB(pdb_filename, str(out_filename), chain)
            try:
                protein = load_structure_np(out_filename,center=False)
                np.save(npy_dir / f"{pdb_id}_{chain}_atomxyz.npy", protein["xyz"])
                np.save(npy_dir / f"{pdb_id}_{chain}_atomtypes.npy", protein["types"])
                np.save(npy_dir / f"{pdb_id}_{chain}_atomflex.npy", protein["flexibility"])
                remove_files_raw(folder_path,pdb_id)
            except Exception:
                print(f"Somethig went wrong with {pdb_id}_{chain}")
                break


if __name__ == '__main__':
    args = parser.parse_args()
    count = 0
    if args.pdb != '':
        pdb_id = args.pdb.split('_')
        chains = pdb_id[1:]
        pdb_id = pdb_id[0]
        get_single(pdb_id, out_dir, chains)
    elif args.pdb_list != '':
        with open(args.pdb_list) as f:
            pdb_list = f.read().splitlines()
        for pdb_id in tqdm(pdb_list, desc='Processing PDBs', unit='pdb'):
           pdb_id = pdb_id.split('_')
           chains = pdb_id[1:]
           pdb_id = pdb_id[0]
           get_single(pdb_id, out_dir, chains)
    else:
        raise ValueError('Must specify PDB or PDB list') 