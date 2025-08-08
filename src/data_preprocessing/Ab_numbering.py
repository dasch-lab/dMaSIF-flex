import pandas as pd
from anarci import number
from tqdm import tqdm

# Chothia regions definition for H and L chains
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

def extract_region(numbered_sequence, chothia_definition):
    """
    Extract the specified regions from the numbered sequence based on Chothia definitions.
    """
    return ''.join(
        [residue for pos, residue in numbered_sequence 
         if any(start <= pos[0] <= end for region in chothia_definition for start, end in [(region['start'], region['end'])])]
    )

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

if __name__ == "__main__":
    file_path = '/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_norep.csv'
    data = pd.read_csv(file_path)
    
    # Process sequences and extract regions
    results_df = process_sequences(data, cdr=True)

    # Write results to CSV
    output_file_path = '/disk1/abtarget/dataset/sabdab/split/sabdab_200423_train1_cdr.csv'
    results_df.to_csv(output_file_path, index=False)

    print(f"Results written to {output_file_path}")
