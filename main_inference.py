# Standard imports:
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path
import pandas as pd

# Custom data loader and model:
from src.data_copy import ProteinPairsSurfaces, PairData, CenterPairAtoms, load_protein_pair
from src.data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from model import dMaSIF
#from old.model_modifications import dMaSIF
from src.data_iteration import iterate
from src.helper import *
from Arguments import parser
import os

os.environ['CMAKE_ARGS'] = (
    '-Wno-dev '
    '-DCMAKE_SUPPRESS_DEVELOPER_WARNINGS=TRUE '
    '-DCMAKE_POLICY_DEFAULT_CMP0146=NEW '
    '-DCMAKE_POLICY_DEFAULT_CMP0148=NEW'
)
os.environ['SKBUILD_CONFIGURE_OPTIONS'] = os.environ['CMAKE_ARGS']

args = parser.parse_args()
args.site = True
args.search = False
args.antibody = True
args.flexibility = True
args.atomic = False
args.binary = False
args.recurrent = True
args.emb_dims = 16
args.n_layers = 5
args.save_emb = False
args.no_chem = False
args.no_geom = False
args.no_flex = False
args.batch_size = 1

if args.search:
    args.pdb_list = "dataset_list/search/Ab_Ag/testing_ppi.txt"
    model_path = "model/search/dmasif_AbAg_atomic_binary_iterflex"
else:
    if args.ab:
        args.pdb_list = "dataset_list/site/test_Ab.txt"
        model_path = "model/site/dmasif_Ab_iterflex"
    else:
        args.pdb_list = "dataset_list/site/test_Ag.txt"
        model_path = "model/site/dmasif_Ag_iterflex"

if args.search:
    prot = 'AbAg'
else:
    if args.ab:
        prot = 'Ab'
    else:
        prot = 'Ag'

folder_data = f"dataset/surface_data_{prot}"

save_predictions_path = Path("preds/" + args.experiment_name)

if args.flexibility:
    args.in_channels = args.in_channels + 1

# Ensure reproducability:
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


# Load the train and test datasets:
transformations = (
    Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else Compose([NormalizeChemFeatures()])
)

if args.single_pdb != "":
    single_data_dir = Path("dataset/raw/01-benchmark_surfaces_npy")
    test_dataset = [load_protein_pair(args.single_pdb, single_data_dir,single_pdb=True, flexibility=args.flexibility, binary = args.binary, atomic = args.atomic)]
    test_pdb_ids = [args.single_pdb]
elif args.pdb_list != "":
    with open(args.pdb_list) as f:
        pdb_list = f.read().splitlines()

    single_data_dir = Path("dataset/raw/01-benchmark_surfaces_npy")
    test_dataset = []
    test_pdb_ids = []
    for pdb in pdb_list:
        try:
            test_dataset.append(load_protein_pair(pdb, single_data_dir,single_pdb=False, flexibility=args.flexibility, binary = args.binary, atomic = args.atomic, itsflexible = args.itsflexible))
            test_pdb_ids.append(pdb)
        except FileNotFoundError:
                continue
        except ValueError:
            continue
else:
    if args.flexibility:
        test_dataset = ProteinPairsSurfaces(
            "surface_data", train=False, ppi=args.search, transform=transformations, flexibility = args.flexibility
        )
    else:
        test_dataset = ProteinPairsSurfaces(
            "surface_data", train=False, ppi=args.search, transform=transformations,
        )
    test_pdb_ids = (
        np.load(f"{folder_data}/testing_pairs_data_ids.npy")
        if args.site
        else np.load(f"{folder_data}/testing_pairs_data_ids_ppi.npy")
    )

    test_dataset = [
        (data, pdb_id)
        for data, pdb_id in zip(test_dataset, test_pdb_ids)
        if iface_valid_filter(data)
    ]
    test_dataset, test_pdb_ids = list(zip(*test_dataset))


# PyTorch geometric expects an explicit list of "batched variables":
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
)


net = dMaSIF(args)
# net.load_state_dict(torch.load(model_path, map_location=args.device))
net.load_state_dict(
    torch.load(model_path, map_location=args.device)["model_state_dict"]
)
net = net.to(args.device)

# Perform one pass through the data:
info = iterate(
    net,
    test_loader,
    None,
    args,
    test=True,
    save_path=save_predictions_path,
    pdb_ids=test_pdb_ids,
    flex = args.flexibility
)


data = {
    "PDB": test_pdb_ids,
    "Accuracy": info['Accuracy'],
    "MCC": info['MCC'],
    "ROC_AUC": info['ROC-AUC'],
    "PR_AUC": info['PR-AUC'],
    
}

#df = pd.DataFrame(data)
#df.to_csv("results_original_search.csv", index=False)

#print(info)
print(f"Accuracy: {np.mean(info['Accuracy'])}")
print(f"MCC: {np.mean(info['MCC'])} pm {np.std(info['MCC'])}")
print(f"ROC-AUC: {np.mean(info['ROC-AUC'])} pm {np.std(info['ROC-AUC'])}")
print(f"PR-AUC: {np.mean(info['PR-AUC'])} pm {np.std(info['PR-AUC'])}")