import numpy as np
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def load_surface_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""

    # Load the data, and read the connectivity information:
    plydata = PlyData.read(str(fname))
    triangles = np.vstack(plydata["face"].data["vertex_indices"])

    # Normalize the point cloud, as specified by the user:
    points = np.vstack([[v[0], v[1], v[2]] for v in plydata["vertex"]])
    if center:
        points = points - np.mean(points, axis=0, keepdims=True)

    nx = plydata["vertex"]["nx"]
    ny = plydata["vertex"]["ny"]
    nz = plydata["vertex"]["nz"]
    normals = np.stack([nx, ny, nz]).T

    # Interface labels
    iface_labels = plydata["vertex"]["iface"]

    # Features
    charge = plydata["vertex"]["charge"]
    hbond = plydata["vertex"]["hbond"]
    hphob = plydata["vertex"]["hphob"]
    features = np.stack([charge, hbond, hphob]).T

    return {
        "xyz": points,
        "triangles": triangles,
        "features": features,
        "iface_labels": iface_labels,
        "normals": normals,
    }


def convert_plys(ply_dir, npy_dir):
    print("Converting PLYs")
    for p in tqdm(ply_dir.glob("*.ply")):
        protein = load_surface_np(p, center=False)
        np.save(npy_dir / (p.stem + "_xyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_triangles.npy"), protein["triangles"])
        np.save(npy_dir / (p.stem + "_features.npy"), protein["features"])
        np.save(npy_dir / (p.stem + "_iface_labels.npy"), protein["iface_labels"])
        np.save(npy_dir / (p.stem + "_normals.npy"), protein["normals"])


def convert_plys(ply_dir, npy_dir, txt):
    print("Converting PLYs")
    for p in tqdm(ply_dir.glob("*.ply")):
        protein = load_surface_np(p, center=False)
        np.save(npy_dir / (p.stem + "_xyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_triangles.npy"), protein["triangles"])
        np.save(npy_dir / (p.stem + "_features.npy"), protein["features"])
        np.save(npy_dir / (p.stem + "_iface_labels.npy"), protein["iface_labels"])
        np.save(npy_dir / (p.stem + "_normals.npy"), protein["normals"])

def convert_plys_txt(ply_dir, npy_dir, txt_file_path):
    print("Converting PLYs")

    # Ensure the output directory exists
    npy_dir.mkdir(parents=True, exist_ok=True)

    # Read the list of PDB entries from the text file
    with open(txt_file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                parts = stripped_line.split('_')
                if len(parts) == 3:
                    pdbid, chainAg, chainAb = parts
                    # Generate the two PLY filenames
                    ply_files = [
                        f"{pdbid}_{chainAg}.ply",
                        f"{pdbid}_{chainAb}.ply"
                    ]
                    for ply_filename in ply_files:
                        ply_path = ply_dir / ply_filename
                        if ply_path.is_file():
                            protein = load_surface_np(ply_path, center=False)
                            np.save(npy_dir / (ply_path.stem + "_xyz.npy"), protein["xyz"])
                            np.save(npy_dir / (ply_path.stem + "_triangles.npy"), protein["triangles"])
                            np.save(npy_dir / (ply_path.stem + "_features.npy"), protein["features"])
                            np.save(npy_dir / (ply_path.stem + "_iface_labels.npy"), protein["iface_labels"])
                            np.save(npy_dir / (ply_path.stem + "_normals.npy"), protein["normals"])
                        else:
                            print(f"File not found: {ply_path}")
                else:
                    print(f"Skipping malformed line: {stripped_line}")


#convert_plys(Path("/disk1/flexibility/dataset/data_preparation_gep/processed"), Path("/disk1/flexibility/dataset/npy"))
#convert_plys_txt(Path("/disk1/fingerprint/data_preparation/01-benchmark_surfaces"), Path("/disk1/fingerprint/SAbDab_preparation/all_structures/npys_tot"), Path("/disk1/flexibility/dataset/Ab_Ag/pdb_antigen_HchainLchain_affinity_missing.txt"))