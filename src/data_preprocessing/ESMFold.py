import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from Bio.SeqIO.PdbIO import BiopythonParserWarning
import warnings
warnings.filterwarnings("ignore", message="'HEADER' line not found", category=BiopythonParserWarning)

# Create the model
is_cuda = torch.cuda.is_available()
# model = esm.pretrained.esmfold_v1()
# model = model.eval().cuda()
from Bio.PDB import PDBParser
from Bio import SeqIO
import pandas as pd
import os
from Bio.SeqUtils import seq1
from anarci import number
from tqdm import tqdm

esm_model = None
tokenizer = None

chothia_H_definition = [
    {"name": "FRH1", "start": 1, "end": 25},
    {"name": "CDRH1", "start": 26, "end": 32},
    {"name": "FRH2", "start": 33, "end": 52},
    {"name": "CDRH2", "start": 53, "end": 55},
    {"name": "FRH3", "start": 56, "end": 95},
    {"name": "CDRH3", "start": 96, "end": 101},
    {"name": "FRH4", "start": 102, "end": 113},
]

chothia_L_definition = [
    {"name": "FRL1", "start": 1, "end": 25},
    {"name": "CDRL1", "start": 26, "end": 32},
    {"name": "FRL2", "start": 33, "end": 49},
    {"name": "CDRL2", "start": 50, "end": 52},
    {"name": "FRL3", "start": 53, "end": 90},
    {"name": "CDRL3", "start": 91, "end": 96},
    {"name": "FRL4", "start": 97, "end": 109},
]

chothia_H_cdrs = [
    {"name": "CDRH1", "start": 26, "end": 32},
    {"name": "CDRH2", "start": 53, "end": 55},
    {"name": "CDRH3", "start": 96, "end": 101},
    
]

chothia_L_cdrs = [
    {"name": "CDRL1", "start": 26, "end": 32},
    {"name": "CDRL2", "start": 50, "end": 52},
    {"name": "CDRL3", "start": 91, "end": 96},
]

"""def extract_region(numbered_sequence, chothia_definition):"""
    #Extract the specified regions from the numbered sequence based on Chothia definitions.
"""
return ''.join(
    [residue for pos, residue in numbered_sequence 
        if any(start <= pos[0] <= end for region in chothia_definition for start, end in [(region['start'], region['end'])])]
)"""



def extract_region(numbered_sequence, chothia_definition):
    """
    Extract the specified regions from the numbered sequence based on Chothia definitions,
    while keeping track of the numbering of each amino acid, up to a maximum position.

    Args:
        numbered_sequence: A list of tuples, where each tuple contains the position and the residue (e.g., [(1, 'A'), (2, 'C'), ...]).
        chothia_definition: A list of dictionaries defining CDR regions with 'start' and 'end' positions.
        max_position: The maximum position to consider in the sequence extraction.

    Returns:
        A tuple containing:
        - extracted_sequence: The amino acid sequence for the specified regions.
        - extracted_positions: A list of positions corresponding to the extracted amino acids.
    """
    extracted_sequence = []
    extracted_positions = []
    
    for pos, residue in numbered_sequence:
        # Check if the position is within the max_position and falls within any of the specified CDR regions
        if any(start <= pos[0] <= end for region in chothia_definition for start, end in [(region['start'], region['end'])]):
            extracted_sequence.append(residue)
            extracted_positions.append(pos[0])  # Add the position (1-based)
    
    #print(len(extracted_sequence))
    
    return ''.join(extracted_sequence).replace('-','X'), extracted_positions



def process_sequences(data, cdr=True):
    """
    Process all VH and VL sequences and extract the regions as specified by Chothia.
    """
    results = []
    
    # Pre-select the dictionary based on the CDR flag
    dictionary_H = chothia_H_cdrs if cdr else chothia_H_definition
    dictionary_L = chothia_L_cdrs if cdr else chothia_L_definition
    
    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing Sequences"):
        pdb_id, vh_sequence, vl_sequence, target = row['name'], row['VH'], row['VL'], row['target']
        label = row.get('label', None)
        
        # Proceed based on sequence length only if not CDR-based
        proceed = cdr or len(vh_sequence) + len(vl_sequence) <= 300

        if proceed:
            try:
                output_vh = number(vh_sequence, scheme='chothia')
                output_vl = number(vl_sequence, scheme='chothia')

                # Extract the regions
                vh = extract_region(output_vh[0], dictionary_H)
                vl = extract_region(output_vl[0], dictionary_L)
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                continue
        else:
            vh, vl = vh_sequence, vl_sequence

        # Append results
        result = {'name': pdb_id, 'VH': vh, 'VL': vl, 'target': target}
        if label is not None:
            result['label'] = label
        results.append(result)

    return pd.DataFrame(results)

def init_model():

  global esm_model
  global tokenizer
  
  if esm_model is not None:
    return
  
  tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
  esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
  if is_cuda:
    esm_model = esm_model.cuda()

  # enable TensorFloat32 computation for a general speedup
  torch.backends.cuda.matmul.allow_tf32 = True

  # Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
  esm_model.trunk.set_chunk_size(64)

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


def init_model():

  global esm_model
  global tokenizer
  
  if esm_model is not None:
    return
  
  tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
  esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
  if is_cuda:
    esm_model = esm_model.cuda()
  
  esm_model.esm = esm_model.esm.half()

  # enable TensorFloat32 computation for a general speedup
  torch.backends.cuda.matmul.allow_tf32 = True

  # Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
  esm_model.trunk.set_chunk_size(64)

class ESMModel():
    def generate_model(self, chain, data, pdb_write=True, model_path=None, anarci_numbering = False):
        init_model()
        plddt = []
        
        # Tokenized input is on CPU by default
        tokenized_input = tokenizer(data, return_tensors="pt", add_special_tokens=False)['input_ids']
        if is_cuda:
          tokenized_input = tokenized_input.cuda()
    
        # Predict structure
        with torch.no_grad():
            output = esm_model(tokenized_input)
        
        if pdb_write:
            # Extract the model structure
            pdb = ESMModel.convert_outputs_to_pdb(output, chain)
            
            # Store result to file
            with open(model_path, "w") as handle:
              if anarci_numbering:
                  pass
              else:
                pdb_with_modified_chain = ESMModel.modify_chain_id(pdb, chain)
              handle.write(pdb_with_modified_chain)

              #handle.write("".join(pdb))
              return True
        else:
            output = {k: v.to("cpu").numpy() for k, v in output.items()}
            plddt.append(output["plddt"])
            return plddt
            
        
    @staticmethod
    def convert_outputs_to_pdb(outputs, chain):
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))       
        return pdbs
    
    def modify_chain_id(pdb, new_chain_id):
      lines = pdb[0].split('\n')
      modified_lines = []
      for line in lines:
          if line.startswith('ATOM') or line.startswith('HETATM'):
              line = line[:21] + new_chain_id + line[22:]
          modified_lines.append(line)
      return '\n'.join(modified_lines) + '\n'
    
    def anarci_numbered_id(pdb, new_chain_id):
        lines = pdb[0].split('\n')
        modified_lines = []
        for line in lines:
          if line.startswith('ATOM') or line.startswith('HETATM'):
              line = line[:21] + new_chain_id + line[22:]
          modified_lines.append(line)
        return '\n'.join(modified_lines) + '\n'

    def get_plddt(output):
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        print(output)
        return output["plddt"]

def parse_pdb_b_factors(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb_structure', pdb_file)
    b_factors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    b_factors.append(atom.get_bfactor())
    return b_factors

def dictionary_pdb_bfactors(structure, b_factor_dict = None):
   if b_factor_dict is None:
       b_factor_dict = {}
   
   for atom in structure.get_atoms():
       atom_name = atom.get_name()
       residue_name = atom.get_parent().get_resname()
       chain = atom.get_parent().get_parent().id
       residue_id = atom.get_parent().get_id()[1]
       b_factor = atom.get_bfactor()
       key = chain + '_' + str(residue_id) + '_' + residue_name + '_' + atom_name
       b_factor_dict[key] = atom.get_bfactor()
   return b_factor_dict

def parse_pdb_b_factors_mean(structure, anarci_numbering = None, b_factors = None):
    if b_factors is None:
      b_factors = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                """if len(anarci_numbering)!=0:# is not None:
                    residue_id = residue.id[1]
                    anarci_index = anarci_numbering[residue_id-1]
                else:"""
                residue_id = residue.id[1]
                b_factor_sum = 0.0
                num_atoms = 0
                for atom in residue:
                    b_factor_sum += atom.get_bfactor()
                    num_atoms += 1
                if num_atoms > 0:
                    mean_b_factor = b_factor_sum / num_atoms
                    """if len(anarci_numbering)!=0:
                        key = (chain.id, residue_id, anarci_index, residue.resname)
                    else:"""
                    key = (chain.id, residue_id, residue.resname)
                    b_factors[key] = mean_b_factor
    return b_factors

def parse_pdb_b_factors_binary(structure, b_factors = None):
    b_factors = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.id[1]
                b_factor_sum = 0.0
                num_atoms = 0
                for atom in residue:
                    b_factor_sum += atom.get_bfactor()
                    num_atoms += 1
                if num_atoms > 0:
                    mean_b_factor = b_factor_sum / num_atoms
                    if 100*mean_b_factor > 80:
                        b_factor = 1
                    else:
                        b_factor = 0
                    key = (chain.id, residue_id, residue.resname)
                    b_factors[key] = b_factor
    return b_factors

def parse_pdb_b_factors_atomic(structure, b_factors = None):
    if b_factors is None:
      b_factors = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.id[1]
                for atom in residue:
                    atom_id = atom.get_id()
                    key = (chain.id, residue_id, residue.resname, atom_id)
                    b_factors[key] = atom.get_bfactor()
    return b_factors


def process_pdbs_from_file(file_path, pdb_folder):
    results = []

    # Read the sampled PDB names from the file
    with open(file_path, 'r') as file:
        pdb_entries = file.readlines()

    # Process each pdb entry
    for entry in pdb_entries[:1]:
        entry = entry.strip()
        pdb_name, chains = entry.split("_")
        
        first_chain, second_chain = chains[0], chains[1]
        pdb_path = os.path.join(pdb_folder, f"{pdb_name}.pdb")
        
        if not os.path.exists(pdb_path):
            print(f"PDB file {pdb_path} not found.")
            continue
        
        # Process both chains
        chain_results = []
        for chain in [first_chain, second_chain]:
            sequence, plddt_tuples = extract_sequence_and_predict(pdb_path, chain)
            chain_results.append(plddt_tuples)

        # Add results to the DataFrame
        results.append({
            "pdb_name": pdb_name,
            "first_chain_result": chain_results[0],
            "second_chain_result": chain_results[1]
        })

    # Convert to a DataFrame
    df = pd.DataFrame(results)
    df.to_csv('predicted_plddt_results.csv', index=False)
    print("Results saved to predicted_plddt_results.csv")


def extract_sequence_and_predict(pdb_file, chain, anarci_numbering = None):
    # Load the structure using Bio.PDB
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file, pdb_file)

    # Extract sequence for the given chain
    sequence = ""
    for model in structure:
        for ch in model:
            if ch.id == chain:
                for residue in ch:
                    if residue.has_id("CA"):  # Alpha carbon check for standard amino acids
                        sequence += seq1(residue.resname)

    # Predict structure with ESMFold
    #model = ESMModel()
    #output_pdb_path = f"{pdb_file.replace('.pdb', f'_{chain}_ESMFold.pdb')}"
    #plddt_values = model.generate_model(chain=chain, data=sequence, pdb_write=True, model_path=output_pdb_path)

    # Parse the predicted PDB and calculate mean pLDDT values
    plddt_tuples = parse_pdb_b_factors_mean(structure, anarci_numbering)
    return sequence, plddt_tuples

def mapp_anarci(anarci_index, sequence):
    map_dict = {}
    for i in range(len(sequence)):
        map_dict[i] = anarci_index[i]
    return map_dict


"""if __name__ == "__main__":
    model = ESMModel()
    sequence_dict = {}
    file_name = "/disk1/fingerprint/sampled_output.txt"
    with open(file_name, 'r') as file:
        lines = file.readlines()

    list_files = [line.strip() for line in lines]
    results = {}
    count = 0
    vh_plddt_dict = {}
    vl_plddt_dict = {}


    for el in tqdm(list_files, desc="Processing PDB files"):
        try:
            pdb_id = el.split('_')[0]
            sequences = [(record.id[-1], str(record.seq)) for record in SeqIO.parse("/disk1/fingerprint/SAbDab_preparation/surface_data_antibody/raw/01-benchmark_pdbs/"+el+".pdb", "pdb-atom")]
            output_vh = number(sequences[0][1], scheme='chothia')
            output_vl = number(sequences[1][1], scheme='chothia')
            vh, order_vh = extract_region(output_vh[0], chothia_H_definition)
            vl, order_vl = extract_region(output_vl[0], chothia_L_definition)
            vh_anarci_mapping = mapp_anarci(order_vh, vh)
            vl_anarci_mapping = mapp_anarci(order_vl, vl)
            out_file_h = el.split('_')[0]+ 'H_ESMFold.pdb'
            path_h = '/'.join(["/disk1/fingerprint/provaESMFold", out_file_h])
            out_file_l = el.split('_')[0]+ 'L_ESMFold.pdb'
            path_l = '/'.join(["/disk1/fingerprint/provaESMFold", out_file_l])
            output_h = model.generate_model(chain = 'H', data=vh, pdb_write=True, model_path=path_h)
            output_l = model.generate_model(chain = 'L', data=vl, pdb_write=True, model_path=path_l)
            sequence_h, plddt_tuples_h = extract_sequence_and_predict(path_h, 'H', vh_anarci_mapping, vh)
            sequence_l, plddt_tuples_l = extract_sequence_and_predict(path_l, 'L', vl_anarci_mapping, vl)

            vh_plddt_dict[pdb_id] = list(sorted(plddt_tuples_h.items()))
            vl_plddt_dict[pdb_id] = list(sorted(plddt_tuples_l.items()))
        except:
            count = count +1
            print(f'Issue with: {el}')
    df_vh = pd.DataFrame({
        'pdbid': list(vh_plddt_dict.keys()),
        'plddtlist': [vh_plddt_dict[pdbid] for pdbid in vh_plddt_dict]
    })
    df_vh.to_csv('vh_plddt_results_more.csv', index=False)

    # Save the VL results to a CSV file in the format pdbid, [(resnum, plddt), ...]
    df_vl = pd.DataFrame({
        'pdbid': list(vl_plddt_dict.keys()),
        'plddtlist': [vl_plddt_dict[pdbid] for pdbid in vl_plddt_dict]
    })
    df_vl.to_csv('vl_plddt_results_more.csv', index=False)

    print(count)"""