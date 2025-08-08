import torch
#from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import numpy as np
from scipy.spatial.transform import Rotation
import math
import urllib.request
import tarfile
from pathlib import Path
import requests
from src.data_preprocessing.convert_pdb2npy import convert_pdbs, convert_pdbs_Ab
from src.data_preprocessing.convert_ply2npy import convert_plys

tensor = torch.FloatTensor
inttensor = torch.LongTensor


def numpy(x):
    return x.detach().cpu().numpy()


def iface_valid_filter(protein_pair):
    labels1 = protein_pair.y_p1.reshape(-1)
    labels2 = protein_pair.y_p2.reshape(-1)
    valid1 = (
        (torch.sum(labels1) < 0.75 * len(labels1))
        and (torch.sum(labels1) > 30)
        and (torch.sum(labels1) > 0.01 * labels2.shape[0])
    )
    valid2 = (
        (torch.sum(labels2) < 0.75 * len(labels2))
        and (torch.sum(labels2) > 30)
        and (torch.sum(labels2) > 0.01 * labels1.shape[0])
    )

    return valid1 and valid2


class RandomRotationPairAtoms(object):
    r"""Randomly rotate a protein"""

    def __call__(self, data):
        R1 = tensor(Rotation.random().as_matrix())
        R2 = tensor(Rotation.random().as_matrix())

        data.atom_coords_p1 = torch.matmul(R1, data.atom_coords_p1.T).T
        data.xyz_p1 = torch.matmul(R1, data.xyz_p1.T).T
        data.normals_p1 = torch.matmul(R1, data.normals_p1.T).T

        data.atom_coords_p2 = torch.matmul(R2, data.atom_coords_p2.T).T
        data.xyz_p2 = torch.matmul(R2, data.xyz_p2.T).T
        data.normals_p2 = torch.matmul(R2, data.normals_p2.T).T

        data.rand_rot1 = R1
        data.rand_rot2 = R2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class CenterPairAtoms(object):
    r"""Centers a protein"""

    def __call__(self, data):
        atom_center1 = data.atom_coords_p1.mean(dim=-2, keepdim=True)
        atom_center2 = data.atom_coords_p2.mean(dim=-2, keepdim=True)

        data.atom_coords_p1 = data.atom_coords_p1 - atom_center1
        data.atom_coords_p2 = data.atom_coords_p2 - atom_center2

        data.xyz_p1 = data.xyz_p1 - atom_center1
        data.xyz_p2 = data.xyz_p2 - atom_center2

        data.atom_center1 = atom_center1
        data.atom_center2 = atom_center2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeChemFeatures(object):
    r"""Centers a protein"""

    def __call__(self, data):
        pb_upper = 3.0
        pb_lower = -3.0

        chem_p1 = data.chemical_features_p1
        chem_p2 = data.chemical_features_p2

        pb_p1 = chem_p1[:, 0]
        pb_p2 = chem_p2[:, 0]
        hb_p1 = chem_p1[:, 1]
        hb_p2 = chem_p2[:, 1]
        hp_p1 = chem_p1[:, 2]
        hp_p2 = chem_p2[:, 2]

        # Normalize PB
        pb_p1 = torch.clamp(pb_p1, pb_lower, pb_upper)
        pb_p1 = (pb_p1 - pb_lower) / (pb_upper - pb_lower)
        pb_p1 = 2 * pb_p1 - 1

        pb_p2 = torch.clamp(pb_p2, pb_lower, pb_upper)
        pb_p2 = (pb_p2 - pb_lower) / (pb_upper - pb_lower)
        pb_p2 = 2 * pb_p2 - 1

        # Normalize HP
        hp_p1 = hp_p1 / 4.5
        hp_p2 = hp_p2 / 4.5

        data.chemical_features_p1 = torch.stack([pb_p1, hb_p1, hp_p1]).T
        data.chemical_features_p2 = torch.stack([pb_p2, hb_p2, hp_p2]).T

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def load_protein_npy(pdb_id, data_dir, center=False, single_pdb=False, flexibility=False, itsflexible = False, atomic = False):
    """Loads a protein surface mesh and its features"""

    # Load the data, and read the connectivity information:
    triangles = (
        None
        if single_pdb
        else inttensor(np.load(data_dir / (pdb_id + "_triangles.npy"))).T
    )
    # Normalize the point cloud, as specified by the user:
    points = None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_xyz.npy")))
    center_location = None if single_pdb else torch.mean(points, axis=0, keepdims=True)

    atom_coords = tensor(np.load(data_dir / (pdb_id + "_atomxyz.npy")))
    atom_types = tensor(np.load(data_dir / (pdb_id + "_atomtypes.npy")))

    if flexibility:
        if itsflexible: 
            atom_flex = 100*tensor(np.load(data_dir / (pdb_id + "_itsflexible.npy")))
            
        if atomic:      
            atom_flex = 100*tensor(np.load(data_dir / (pdb_id + "_atomflex_atom.npy")))
        else:
            atom_flex = 100*tensor(np.load(data_dir / (pdb_id + "_atomflex.npy")))
            
        atom_flex = atom_flex.reshape(-1, 1)

    else:
        atom_flex = None

    if center:
        points = points - center_location
        atom_coords = atom_coords - center_location

    # Interface labels
    iface_labels = (
        None
        if single_pdb
        else tensor(np.load(data_dir / (pdb_id + "_iface_labels.npy")).reshape((-1, 1)))
    )

    # Features
    chemical_features = (
        None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_features.npy")))
    )

    # Normals
    normals = (
        None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_normals.npy")))
    )

    if flexibility:
        protein_data = Data(
            xyz=points,
            face=triangles,
            chemical_features=chemical_features,
            y=iface_labels,
            normals=normals,
            center_location=center_location,
            num_nodes=None if single_pdb else points.shape[0],
            atom_coords=atom_coords,
            atom_types=atom_types,
            atom_flexibility = atom_flex,
        )
    else:
        protein_data = Data(
            xyz=points,
            face=triangles,
            chemical_features=chemical_features,
            y=iface_labels,
            normals=normals,
            center_location=center_location,
            num_nodes=None if single_pdb else points.shape[0],
            atom_coords=atom_coords,
            atom_types=atom_types,
        )
    return protein_data

class PairData(Data): #TODO: deal flexibility
    def __init__(
        self,
        xyz_p1=None,
        xyz_p2=None,
        face_p1=None,
        face_p2=None,
        chemical_features_p1=None,
        chemical_features_p2=None,
        y_p1=None,
        y_p2=None,
        normals_p1=None,
        normals_p2=None,
        center_location_p1=None,
        center_location_p2=None,
        atom_coords_p1=None,
        atom_coords_p2=None,
        atom_types_p1=None,
        atom_types_p2=None,
        atom_center1=None,
        atom_center2=None,
        rand_rot1=None,
        rand_rot2=None,
        atom_flex1=None,
        atom_flex2=None,
    ):
        super().__init__()
        self.xyz_p1 = xyz_p1
        self.xyz_p2 = xyz_p2
        self.face_p1 = face_p1
        self.face_p2 = face_p2

        self.chemical_features_p1 = chemical_features_p1
        self.chemical_features_p2 = chemical_features_p2
        self.y_p1 = y_p1
        self.y_p2 = y_p2
        self.normals_p1 = normals_p1
        self.normals_p2 = normals_p2
        self.center_location_p1 = center_location_p1
        self.center_location_p2 = center_location_p2
        self.atom_coords_p1 = atom_coords_p1
        self.atom_coords_p2 = atom_coords_p2
        self.atom_types_p1 = atom_types_p1
        self.atom_types_p2 = atom_types_p2
        self.atom_center1 = atom_center1
        self.atom_center2 = atom_center2
        self.rand_rot1 = rand_rot1
        self.rand_rot2 = rand_rot2
        self.atom_flexibility1=atom_flex1
        self.atom_flexibility2=atom_flex2
     

    def __inc__(self, key, value, store):
        if key == "face_p1":
            return self.xyz_p1.size(0)
        if key == "face_p2":
            return self.xyz_p2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

    def __cat_dim__(self, key, value, store):
        if ("index" in key) or ("face" in key):
            return 1
        else:
            return 0

def load_protein_pair(pdb_id, data_dir,single_pdb=False, flexibility=False, itsflexible = False, atomic = False):
    """Loads a protein surface mesh and its features"""
    pspl = pdb_id.split("_")
    p1_id = pspl[0] + "_" + pspl[1]
    p2_id = pspl[0] + "_" + pspl[2]

    try:
        p1 = load_protein_npy(p1_id, data_dir, center=False, single_pdb=single_pdb, flexibility = flexibility, itsflexible = itsflexible, atomic = atomic)
    except FileNotFoundError:
        print(f"Issue with {p1_id}")
        raise  # Reraise the exception to propagate it further
    except ValueError as e:
        print("Issue with flexibility {}: {}".format(p1_id, str(e)))
        raise
    
    try:
        p2 = load_protein_npy(p2_id, data_dir, center=False, single_pdb=single_pdb, flexibility = flexibility, itsflexible = itsflexible, atomic = atomic)
    except FileNotFoundError:
        print(f"Issue with {p2_id}")
        raise
    except ValueError as e:
        print("Issue with flexibility {}: {}".format(p2_id, str(e)))
        raise

    if flexibility:
        try:
            atom_flex_p1=p1["atom_flexibility"]
            atom_flex_p2=p2["atom_flexibility"]
        except FileNotFoundError:
            print('notfound')
    else:
        atom_flex_p1=None
        atom_flex_p2=None


    # pdist = ((p1['xyz'][:,None,:]-p2['xyz'][None,:,:])**2).sum(-1).sqrt()
    # pdist = pdist<2.0
    # y_p1 = (pdist.sum(1)>0).to(torch.float).reshape(-1,1)
    # y_p2 = (pdist.sum(0)>0).to(torch.float).reshape(-1,1)

    y_p1 = p1["y"]
    y_p2 = p2["y"]


    if flexibility:
        protein_pair_data = PairData(
            xyz_p1=p1["xyz"],
            xyz_p2=p2["xyz"],
            face_p1=p1["face"],
            face_p2=p2["face"],
            chemical_features_p1=p1["chemical_features"],
            chemical_features_p2=p2["chemical_features"],
            y_p1=y_p1,
            y_p2=y_p2,
            normals_p1=p1["normals"],
            normals_p2=p2["normals"],
            center_location_p1=p1["center_location"],
            center_location_p2=p2["center_location"],
            atom_coords_p1=p1["atom_coords"],
            atom_coords_p2=p2["atom_coords"],
            atom_types_p1=p1["atom_types"],
            atom_types_p2=p2["atom_types"],
            atom_flex1=atom_flex_p1,
            atom_flex2=atom_flex_p2,
        )
    else:
        protein_pair_data = PairData(
            xyz_p1=p1["xyz"],
            xyz_p2=p2["xyz"],
            face_p1=p1["face"],
            face_p2=p2["face"],
            chemical_features_p1=p1["chemical_features"],
            chemical_features_p2=p2["chemical_features"],
            y_p1=y_p1,
            y_p2=y_p2,
            normals_p1=p1["normals"],
            normals_p2=p2["normals"],
            center_location_p1=p1["center_location"],
            center_location_p2=p2["center_location"],
            atom_coords_p1=p1["atom_coords"],
            atom_coords_p2=p2["atom_coords"],
            atom_types_p1=p1["atom_types"],
            atom_types_p2=p2["atom_types"],
        )

    return protein_pair_data
    


class ProteinPairsSurfaces(InMemoryDataset):
    url = ""

    def __init__(self, root, ppi=False, train=True, transform=None, pre_transform=None, flexibility=False, antibody = False, mix = False, itsflexible=False, atomic = False):
        self.ppi = ppi
        self.antibody = antibody
        self.flexibility = flexibility
        self.mix = mix
        self.itsflexible = itsflexible
        self.atomic = atomic
        super(ProteinPairsSurfaces, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        if not self.ppi:
            file_names = [
                "training_pairs_data.pt",
                "testing_pairs_data.pt",
                "training_pairs_data_ids.npy",
                "testing_pairs_data_ids.npy",
            ]
        else:
            file_names = [
                "training_pairs_data_ppi.pt",
                "testing_pairs_data_ppi.pt",
                "training_pairs_data_ids_ppi.npy",
                "testing_pairs_data_ids_ppi.npy",
            ]
        return file_names

    def process(self):
        #pdb_dir = Path(self.root) / "raw" / "01-benchmark_pdbs"
        #surf_dir = Path(self.root) / "raw" / "01-benchmark_surfaces"
        #protein_dir = Path(self.root) / "raw" / "01-benchmark_surfaces_npy"
        #lists_dir = Path('./lists')
        pdb_dir = Path(self.root) / "raw" / "01-benchmark_pdbs" 
        surf_dir = Path(self.root) / "raw" / "01-benchmark_surfaces"
        protein_dir = Path(self.root) / "raw" / "01-benchmark_surfaces_npy"
        if self.ppi:
            pass
        else:   
            lists_dir = Path("/disk1/flexibility/dataset/Ab_Ag")
            csv_files = Path("./results_cdr3_extracted_gep.csv")


        # Untar surface files
        if not (pdb_dir.exists() and surf_dir.exists()):
            tar = tarfile.open(self.raw_paths[0])
            tar.extractall(self.raw_dir)
            tar.close()

        with open(lists_dir / "training_Ag.txt") as f_tr, open(
            lists_dir / "validation_Ag.txt"
        ) as f_ts:
            training_list = sorted(f_tr.read().splitlines())
            testing_list = sorted(f_ts.read().splitlines())
    

        with open(lists_dir / "training_abag_original.txt") as f_tr, open(
            lists_dir / "validation_abag.txt"
        ) as f_ts:
            training_pairs_list = sorted(f_tr.read().splitlines())
            testing_pairs_list = sorted(f_ts.read().splitlines())
            pairs_list = sorted(training_pairs_list + testing_pairs_list)
        
        if not protein_dir.exists():
            protein_dir.mkdir(parents=False, exist_ok=False)
            convert_plys(surf_dir,protein_dir)
            if self.antibody:
                convert_pdbs_Ab(pdb_dir,protein_dir, list_names = pairs_list, atomic = self.atomic, itsflexible=self.itsflexible, csv_files = csv_files)
            else:
                convert_pdbs(pdb_dir,protein_dir, list_names = pairs_list, atomic = self.atomic, csv_files = csv_files)

        if not self.ppi:
            training_pairs_list = []
            for p in pairs_list:
                pspl = p.split("_")
                p1 = pspl[0] + "_" + pspl[1]
                p2 = pspl[0] + "_" + pspl[2]

                if p1 in training_list:
                    training_pairs_list.append(p)
                if p2 in training_list:
                    training_pairs_list.append(pspl[0] + "_" + pspl[2] + "_" + pspl[1])

            testing_pairs_list = []
            for p in pairs_list:
                pspl = p.split("_")
                p1 = pspl[0] + "_" + pspl[1]
                p2 = pspl[0] + "_" + pspl[2]
                if p1 in testing_list:
                    testing_pairs_list.append(p)
                if p2 in testing_list:
                    testing_pairs_list.append(pspl[0] + "_" + pspl[2] + "_" + pspl[1])

        # # Read data into huge `Data` list.
        training_pairs_data = []
        training_pairs_data_ids = []
        for p in training_pairs_list:
            try:
                protein_pair = load_protein_pair(p, protein_dir, flexibility=self.flexibility, itsflexible = self.itsflexible, atomic = self.atomic)
            except FileNotFoundError:
                continue
            except ValueError:
                continue
            training_pairs_data.append(protein_pair)
            training_pairs_data_ids.append(p)

        testing_pairs_data = []
        testing_pairs_data_ids = []
        for p in testing_pairs_list:
            try:
                protein_pair = load_protein_pair(p, protein_dir, flexibility=self.flexibility, itsflexible = self.itsflexible, atomic = self.atomic)
            except FileNotFoundError:
                continue
            except ValueError:
                continue
            testing_pairs_data.append(protein_pair)
            testing_pairs_data_ids.append(p)

        if self.pre_filter is not None:
            training_pairs_data = [
                data for data in training_pairs_data if self.pre_filter(data)
            ]
            testing_pairs_data = [
                data for data in testing_pairs_data if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            training_pairs_data = [
                self.pre_transform(data) for data in training_pairs_data
            ]
            testing_pairs_data = [
                self.pre_transform(data) for data in testing_pairs_data
            ]

        print(testing_pairs_data)
        training_pairs_data, training_pairs_slices = self.collate(training_pairs_data)
        torch.save(
            (training_pairs_data, training_pairs_slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[2], training_pairs_data_ids)
        testing_pairs_data, testing_pairs_slices = self.collate(testing_pairs_data)
        torch.save((testing_pairs_data, testing_pairs_slices), self.processed_paths[1])
        np.save(self.processed_paths[3], testing_pairs_data_ids)