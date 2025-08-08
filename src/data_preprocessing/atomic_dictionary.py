"""hydrogen_to_heavy = {
    "GLY": {
        'H': 'N', 'HA2': 'CA', 'HA3': 'CA'
    },
    "ALA": {
        'H': 'N', 'HA': 'CA', 'HB1': 'CB', 'HB2': 'CB', 'HB3': 'CB'
    },
    "VAL": {
        'H': 'N', 'HA': 'CA', 'HB': 'CB', 
        'HG11': 'CG1', 'HG12': 'CG1', 'HG13': 'CG1', 
        'HG21': 'CG2', 'HG22': 'CG2', 'HG23': 'CG2'
    },
    "LEU": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 'HG': 'CG', 
        'HD11': 'CD1', 'HD12': 'CD1', 'HD13': 'CD1', 
        'HD21': 'CD2', 'HD22': 'CD2', 'HD23': 'CD2'
    },
    "ILE": {
        'H': 'N', 'HA': 'CA', 'HB': 'CB', 
        'HG12': 'CG1', 'HG13': 'CG1', 
        'HG21': 'CG2', 'HG22': 'CG2', 'HG23': 'CG2', 
        'HD11': 'CD1', 'HD12': 'CD1', 'HD13': 'CD1'
    },
    "PHE": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD1': 'CD1', 'HD2': 'CD2', 'HE1': 'CE1', 'HE2': 'CE2', 'HZ': 'CZ'
    },
    "TYR": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD1': 'CD1', 'HD2': 'CD2', 'HE1': 'CE1', 'HE2': 'CE2', 'HH': 'OH'
    },
    "TRP": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD1': 'CD1', 'HE1': 'NE1', 'HZ2': 'CZ2', 'HH2': 'CH2', 'HZ3': 'CZ3'
    },
    "SER": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 'HG': 'OG'
    },
    "THR": {
        'H': 'N', 'HA': 'CA', 'HB': 'CB', 
        'HG1': 'OG1', 'HG21': 'CG2', 'HG22': 'CG2', 'HG23': 'CG2'
    },
    "CYS": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 'HG': 'SG'
    },
    "MET": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HE1': 'CE', 'HE2': 'CE', 'HE3': 'CE'
    },
    "ASP": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB'
    },
    "GLU": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG'
    },
    "ASN": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD21': 'ND2', 'HD22': 'ND2'
    },
    "GLN": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HE21': 'NE2', 'HE22': 'NE2'
    },
    "LYS": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HD2': 'CD', 'HD3': 'CD', 
        'HE2': 'CE', 'HE3': 'CE', 
        'HZ1': 'NZ', 'HZ2': 'NZ', 'HZ3': 'NZ'
    },
    "ARG": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HD2': 'CD', 'HD3': 'CD', 
        'HE': 'NE', 
        'HH11': 'NH1', 'HH12': 'NH1', 'HH21': 'NH2', 'HH22': 'NH2'
    },
    "HIS": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD2': 'CD2', 'HE1': 'NE2'
    },
    "PRO": {
        'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HD2': 'CD', 'HD3': 'CD'
    }
}
"""

hydrogen_to_heavy = {
    "GLY": {
        'H': 'N', 'HA2': 'CA', 'HA3': 'CA'
    },
    "ALA": {
        'H': 'N', 'HA': 'CA', 'HB1': 'CB', 'HB2': 'CB', 'HB3': 'CB'
    },
    "VAL": {
        'H': 'N', 'HA': 'CA', 'HB': 'CB', 
        'HG11': 'CG1', 'HG12': 'CG1', 'HG13': 'CG1', 
        'HG21': 'CG2', 'HG22': 'CG2', 'HG23': 'CG2'
    },
    "LEU": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG': 'CG', 
        'HD11': 'CD1', 'HD12': 'CD1', 'HD13': 'CD1', 
        'HD21': 'CD2', 'HD22': 'CD2', 'HD23': 'CD2'
    },
    "ILE": {
        'H': 'N', 'HA': 'CA', 'HB': 'CB', 
        'HG12': 'CG1', 'HG13': 'CG1', 
        'HG21': 'CG2', 'HG22': 'CG2', 'HG23': 'CG2', 
        'HD11': 'CD1', 'HD12': 'CD1', 'HD13': 'CD1'
    },
    "PHE": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD1': 'CD1', 'HD2': 'CD2', 'HE1': 'CE1', 'HE2': 'CE2', 'HZ': 'CZ'
    },
    "TYR": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD1': 'CD1', 'HD2': 'CD2', 'HE1': 'CE1', 'HE2': 'CE2', 'HH': 'OH'
    },
    "TRP": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD1': 'CD1', 'HE1': 'NE1', 'HZ2': 'CZ2', 'HH2': 'CH2', 'HZ3': 'CZ3',
        'HE3': 'CE3'  # Added missing HE3 mapping
    },
    "SER": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 'HG': 'OG'
    },
    "THR": {
        'H': 'N', 'HA': 'CA', 'HB': 'CB', 
        'HG1': 'OG1', 'HG21': 'CG2', 'HG22': 'CG2', 'HG23': 'CG2'
    },
    "CYS": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 'HG': 'SG'
    },
    "MET": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HE1': 'CE', 'HE2': 'CE', 'HE3': 'CE'
    },
    "ASP": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB'
    },
    "GLU": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG'
    },
    "ASN": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD21': 'ND2', 'HD22': 'ND2'
    },
    "GLN": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HE21': 'NE2', 'HE22': 'NE2'
    },
    "LYS": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HD2': 'CD', 'HD3': 'CD', 
        'HE2': 'CE', 'HE3': 'CE', 
        'HZ1': 'NZ', 'HZ2': 'NZ', 'HZ3': 'NZ'
    },
    "ARG": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HD2': 'CD', 'HD3': 'CD', 
        'HE': 'NE', 
        'HH11': 'NH1', 'HH12': 'NH1', 'HH21': 'NH2', 'HH22': 'NH2'
    },
    "HIS": {
        'H': 'N', 'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HD1': 'ND1', 'HD2': 'CD2', 'HE1': 'CE1', 'HE2': 'NE2'  # Added missing HD1 and HE2 mappings
    },
    "PRO": {
        'HA': 'CA', 'HB2': 'CB', 'HB3': 'CB', 
        'HG2': 'CG', 'HG3': 'CG', 'HD2': 'CD', 'HD3': 'CD'
    }
}
