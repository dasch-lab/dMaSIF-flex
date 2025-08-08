import re
import struct
from Bio.PDB import PDBParser, PDBIO
import os

'''
A table to convert three-letters code AAs to one letter
'''
aaTable = {
  'ALA':'A',
  'ARG':'R',
  'ASN':'N',
  'ASP':'D',
  'ASX':'B',
  'CYS':'C',
  'GLU':'E',
  'GLN':'Q',
  'GLX':'Z',
  'GLY':'G',
  'HIS':'H',
  'ILE':'I',
  'LEU':'L',
  'LYS':'K',
  'MET':'M',
  'PHE':'F',
  'PRO':'P',
  'SER':'S',
  'THR':'T',
  'TRP':'W',
  'TYR':'Y',
  'VAL':'V'
}

atom_fields = {
  'record': 6,  # record
  'serial': 5,  # atom serial
  'skip1': -1,
  'name': 4,  # atom name
  'altloc':1,  # altloc
  'resname':3,  # res name
  'skip2': -1,
  'chain':1,  # chain id
  'resnum':4,  # res number
  'achar': 1,  # AChar
  'skip3': -3,
  'x': 8,  # X
  'y': 8,  # Y
  'z': 8,  # Z
  'occupancy': 6,  # Occupancy
  'temp': 6,  # tempFactor
  'skip4': -10,
  'element': 2,  # Element
  'charge': 2   # charge
}

class atom:
  def __init__(self, data):
    self._data = data

  @staticmethod
  def lineParse(line):
    pass

  def toPDB(self):
    name = self._data['name']
    # Check if the atom name is less than four characters long
    if len(name) < 4:
        name = name.center(4)  # Center align the atom name in a field of width 4
    else:
        name = name[:4]  # Truncate the atom name to four characters if it's longer
    
    return 'ATOM  {serial: >5} {name}{altloc: <1}{resname: <3} {chain}{resnum: >4}{achar: <1}   {x: >8}{y: >8}{z: >8}{occupancy: >6}{temp: >6}          {element: >2}{charge: >2}'.format(
        serial=self._data['serial'],
        name=name,
        altloc=self._data['altloc'],
        resname=self._data['resname'],
        chain=self._data['chain'],
        resnum=self._data['resnum'],
        achar=self._data['achar'],
        x=self._data['x'],
        y=self._data['y'],
        z=self._data['z'],
        occupancy=self._data['occupancy'],
        temp=self._data['temp'],
        element=self._data['element'],
        charge=self._data['charge']
    )


  def __getitem__(self, arg):
    if arg in self._data:
      return self._data[arg]
    else:
      return None
    
  def __setitem__(self, arg, value):
    self._data[arg]=value

def sequence(path, chain=None):

  '''
  Extract the sequence from the pdb
  '''

  sequence = ''
  status = {
    'chain': None,
    'resnum': None,
    'resid': None
  }
  for line in parse(path):

    chain = line['chain'] if not chain else chain
    if line['chain'] != chain:
      continue

    # Update sequence
    resid = '{}{}'.format(line['resnum'], line['achar'])
    if status['resid'] != resid:
      sequence += aaTable[ line['resname'] ]
    # if status['resnum'] != line['resnum']:
    #   sequence += aaTable[ line['resname'] ]

    # Update status
    status['resid'] = resid
    status['chain'] = line['chain']
    status['resnum'] = line['resnum']

  return sequence

def parse(path):

  '''
  Parse the PDB line by line and return the parsed atom line
  '''
  formatString = ''.join('{}{}'.format(abs(fw), 'x' if fw < 0 else 's') for fw in atom_fields.values())
  keyList = [ key for key, value in atom_fields.items() if value > 0 ]
  fieldstruct = struct.Struct(formatString)
  unpack = fieldstruct.unpack_from
  atomParse = lambda line: [ s.decode() for s in unpack(line.encode()) ]

  # Process input file
  coord_re = re.compile('^(ATOM)')
  with open(path) as handle:
    for line in handle:

      # Skip everything but coordinate lines
      if not coord_re.match(line):
        continue

      line = line.ljust(80)
      data = atomParse(line)
      data = [ data[i].strip() for i in range(len(data)) ]
      data = dict(zip(keyList, data))
      
      yield atom(data)

def change_res_id(file_name):
  counter = 0
  atom_list = []
  old = 0
  old_chain = None
  old_name = None
  old_achar = None
  for name in parse(file_name):
      if old_chain is None:
        old_chain = name['chain']
      if old_name is None:
         old_name = name['resname']
         print(name['resname'])
         print(name['achar'])
      if old_achar is None:
         old_achar = name['achar']
      
      if old_chain == name['chain']: 
        if old != name['resnum'] or old_name != name['resname'] or old_achar != name['achar']:
          old = name['resnum']
          old_name = name['resname']
          old_achar = name['achar']
          counter += 1
        name['resnum'] = counter
      else:
        old_chain = name['chain']
        counter = 0
        name['resnum'] = counter + 1
      atom_list.append(name)
    
  with open(file_name, 'w') as handle:
    for name in atom_list:
      handle.write(name.toPDB() + '\n')
  return True


def compare_pdb_structures(pdb_filename1, pdb_filename2):
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure(pdb_filename1, pdb_filename1 + ".pdb")
    structure2 = parser.get_structure(pdb_filename2, pdb_filename2 + ".pdb")
    
    comparison_list = []
    chains1 = [chain for chain in structure1.get_chains() if chain.id == 'H']
    chains2 = [chain for chain in structure2.get_chains() if chain.id == 'H']


    # Iterate over chains in both structures
    for chain1, chain2 in zip(chains1, chains2):
        # Iterate over residues in both chains
        for residue1, residue2 in zip(chain1, chain2):
            # Extract residue ID and residue name
            res_id1 = residue1.get_id()[1]
            res_id2 = residue2.get_id()[1]
            res_name1 = residue1.get_resname()
            res_name2 = residue2.get_resname()

            # Compare residue IDs and residue names
            if res_id1 == res_id2 and res_name1 == res_name2:
                comparison_list.append((res_id1, res_name1, res_id2, res_name2, "Identical"))
            else:
                comparison_list.append((res_id1, res_name1, res_id2, res_name2, "Different"))

    return comparison_list


def extract_chains(input_pdb_path, chain_groups, output_folder):
    atom_list = []
    #parser = PDBParser()
    #structure = parser.get_structure(input_pdb_path, input_pdb_path + '.pdb')
    chain_list = [el for el in chain_groups]

    for name in parse(input_pdb_path):
       if name['chain'] in chain_groups:
             atom_list.append(name)
    
    outputfile = output_folder + '/' + input_pdb_path.split('/')[-1].split('.')[0] +'_'+ chain_groups + '.pdb'

    with open(outputfile, 'w') as handle:
      for name in atom_list:
        handle.write(name.toPDB() + '\n')
    return True

"""
#change_res_id("/disk1/fingerprint/data_preparation/00-raw_pdbs/4FQI copy.pdb")
pdb_filename1 = "/disk1/fingerprint/data_preparation/00-raw_pdbs/4FQI"
pdb_filename2 = "/disk1/fingerprint/provaESMFold/4FQI_H_ESMFold"
comparison_result = compare_pdb_structures(pdb_filename1, pdb_filename2)
for item in comparison_result:
    print(item)

#change_res_id("/disk1/fingerprint/tmp/4FQI_AB copy.pdb")
"""