import torch
import numpy as np
from helper import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import math
from tqdm import tqdm
from geometry_processing import save_vtk
from helper import numpy, diagonal_ranges
import time
from pytorch_earlystopping import EarlyStopping
#from model import dMaSIF
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
import os



def process_single(protein_pair, chain_idx=1, flex = False):
    """Turn the PyG data object into a dict."""
    #print(">>>", type(protein_pair))
    P = {}
    # with_mesh = "face_p1" in protein_pair.keys
    # preprocessed = "gen_xyz_p1" in protein_pair.keys
    with_mesh = hasattr(protein_pair, "face_p1")
    preprocessed = hasattr(protein_pair, "gen_xyz_p1")

    if chain_idx == 1:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p1 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p1_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p1 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p1 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p1
        P["batch_atoms"] = protein_pair.atom_coords_p1_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p1
        P["atomtypes"] = protein_pair.atom_types_p1
        if flex:  
            P["atomflex"] = protein_pair.atom_flexibility1 

        P["xyz"] = protein_pair.gen_xyz_p1 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p1 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p1 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p1 if preprocessed else None

    elif chain_idx == 2:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p2 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p2_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p2 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p2 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p2
        P["batch_atoms"] = protein_pair.atom_coords_p2_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p2
        P["atomtypes"] = protein_pair.atom_types_p2
        if flex:
            P["atomflex"] = protein_pair.atom_flexibility2

        P["xyz"] = protein_pair.gen_xyz_p2 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p2 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p2 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p2 if preprocessed else None

    return P


def save_protein_batch_single(protein_pair_id, P, save_path, pdb_idx):

    protein_pair_id = protein_pair_id.split("_")
    pdb_id = protein_pair_id[0] + "_" + protein_pair_id[pdb_idx]

    batch = P["batch"]

    xyz = P["xyz"]

    inputs = P["input_features"]

    embedding = P["embedding_1"] if pdb_idx == 1 else P["embedding_2"]
    emb_id = 1 if pdb_idx == 1 else 2

    predictions = torch.sigmoid(P["iface_preds"]) if "iface_preds" in P.keys() else 0.0*embedding[:,0].view(-1, 1)

    labels = P["labels"].view(-1, 1) if P["labels"] is not None else 0.0 * predictions

    coloring = torch.cat([inputs, embedding, predictions, labels], axis=1)

    save_vtk(str(save_path / pdb_id) + f"_pred_emb{emb_id}", xyz, values=coloring)
    np.save(str(save_path / pdb_id) + "_predcoords", numpy(xyz))
    np.save(str(save_path / pdb_id) + f"_predfeatures_emb{emb_id}", numpy(coloring))


def project_iface_labels(P, threshold=2.0):

    queries = P["xyz"]
    batch_queries = P["batch"]
    source = P["mesh_xyz"]
    batch_source = P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1, 1) < threshold
    ).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein

    query_labels = labels[nn_i] * nn_dist_i

    P["labels"] = query_labels


def process(args, protein_pair, net, flex):
    P1 = process_single(protein_pair, chain_idx=1, flex=flex)
    # if not "gen_xyz_p1" in protein_pair.keys:
    if not hasattr(protein_pair, "gen_xyz_p1"):
        net.preprocess_surface(P1, flex)
        #if P1["mesh_labels"] is not None:
        #    project_iface_labels(P1)
    P2 = None
    if not args.single_protein:
        P2 = process_single(protein_pair, chain_idx=2, flex=flex)
        #if not "gen_xyz_p2" in protein_pair.keys:
        if not hasattr(protein_pair, "gen_xyz_p1"):
            net.preprocess_surface(P2, flex)
            #if P2["mesh_labels"] is not None:
            #    project_iface_labels(P2)

    return P1, P2


def generate_matchinglabels(args, P1, P2):
    if args.random_rotation:
        P1["xyz"] = torch.matmul(P1["rand_rot"].T, P1["xyz"].T).T + P1["atom_center"]
        P2["xyz"] = torch.matmul(P2["rand_rot"].T, P2["xyz"].T).T + P2["atom_center"]
    xyz1_i = LazyTensor(P1["xyz"][:, None, :].contiguous())
    xyz2_j = LazyTensor(P2["xyz"][None, :, :].contiguous())

    xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1).sqrt()
    xyz_dists = (1.0 - xyz_dists).step()

    p1_iface_labels = (xyz_dists.sum(1) > 1.0).float().view(-1)
    p2_iface_labels = (xyz_dists.sum(0) > 1.0).float().view(-1)

    P1["labels"] = p1_iface_labels
    P2["labels"] = p2_iface_labels

def compute_proximity_to_tp(TP_xyz, FP_xyz, FN_xyz, threshold_distance=5.0):
    """
    Check if False Positives and False Negatives are nearby True Positives.
    TP_xyz, FP_xyz, FN_xyz should be tensors of shape [n, 3], where n is the number of residues.
    """
    nearby_fp = []
    nearby_fn = []

    # Compute distances between each FP and TP
    for fp in FP_xyz:
        distances = torch.norm(TP_xyz - fp, dim=1)  # Compute Euclidean distance between FP and each TP
        if torch.min(distances) < threshold_distance:
            nearby_fp.append(True)
        else:
            nearby_fp.append(False)

    # Compute distances between each FN and TP
    for fn in FN_xyz:
        distances = torch.norm(TP_xyz - fn, dim=1)  # Compute Euclidean distance between FN and each TP
        if torch.min(distances) < threshold_distance:
            nearby_fn.append(True)
        else:
            nearby_fn.append(False)

    return nearby_fp, nearby_fn


def compute_loss(args, P1, P2, n_points_sample=16, all = False):

    if args.flexibility:
        pos_weight = torch.tensor([10.0])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(args.device))
    #print("-------------------------- HERE -------------------------")

    #print(f'P1: {P1.keys()}')
    #print(f'P2: {P2.keys()}')
    plddt_pos = []
    plddt_neg = []
    flexibility = args.flexibility
    tp_xyz = []
    fp_xyz = []
    fn_xyz = []

    if args.search:
        pos_xyz1 = P1["xyz"][P1["labels"] == 1]
        pos_xyz2 = P2["xyz"][P2["labels"] == 1]
        pos_descs1 = P1["embedding_1"][P1["labels"] == 1]
        pos_descs2 = P2["embedding_2"][P2["labels"] == 1]

        # Compute distances for positive samples
        pos_xyz_dists = ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1).sqrt()
        pos_desc_dists = torch.matmul(pos_descs1, pos_descs2.T)

        # Get pLDDT values as tuples (instead of multiplying)
        try:
            plddt_1_pos = P1["input_features"][P1["labels"] == 1][:, 16].unsqueeze(1)  # Shape: [106, 1]
            plddt_2_pos = P2["input_features"][P2["labels"] == 1][:, 16].unsqueeze(0)  # Shape: [1, 99]
            # Save the pLDDT values as tuples instead of multiplying
            # Shape: [106, 99, 2]
            plddt_descriptors_pos = torch.stack((plddt_1_pos.expand(-1, plddt_2_pos.size(1)),
                                                plddt_2_pos.expand(plddt_1_pos.size(0), -1)), dim=-1)
        except:
            flexibility = False

        

        # Filter based on xyz distance < 1.0
        pos_preds = pos_desc_dists[pos_xyz_dists < 1.0]
        pos_labels = torch.ones_like(pos_preds)
        pos_preds_p1 = pos_preds
        pos_labels_p1 = pos_labels

        if flexibility:
            # Filter pLDDT descriptors based on the same mask
            plddt_descriptors_positive = plddt_descriptors_pos[pos_xyz_dists < 1.0]
            # Shape: [N_filtered, 2]

        # Negative sample descriptors
        n_desc_sample = 100
        sample_desc2_id = torch.randperm(len(P2["embedding_2"]))[:n_desc_sample]
        sample_desc2 = P2["embedding_2"][sample_desc2_id]
        neg_preds = torch.matmul(pos_descs1, sample_desc2.T).view(-1)
        neg_labels = torch.zeros_like(neg_preds)
        neg_preds_p1 = neg_preds
        neg_labels_p1 = neg_labels


        if flexibility:
            # Get pLDDT values for negative samples as tuples
            plddt_1_neg = P1["input_features"][P1["labels"] == 1][:, 16].unsqueeze(1)  # Shape: [106, 1]
            plddt_2_neg = P2["input_features"][sample_desc2_id][:, 16].unsqueeze(0)  # Shape: [1, 100]
            plddt_descriptors_neg = torch.stack((plddt_1_neg.expand(-1, plddt_2_neg.size(1)),
                                                plddt_2_neg.expand(plddt_1_neg.size(0), -1)), dim=-1)
            # Flatten to match the shape of neg_preds
            plddt_descriptors_neg = plddt_descriptors_neg.view(-1, 2)

        # Symmetry in positive samples
        pos_descs1_2 = P1["embedding_2"][P1["labels"] == 1]
        pos_descs2_2 = P2["embedding_1"][P2["labels"] == 1]
        pos_desc_dists2 = torch.matmul(pos_descs2_2, pos_descs1_2.T)

        if flexibility:
            # Get pLDDT values for symmetry
            plddt_1_sym_pos = P2["input_features"][P2["labels"] == 1][:, 16].unsqueeze(1)  # Shape: [99, 1]
            plddt_2_sym_pos = P1["input_features"][P1["labels"] == 1][:, 16].unsqueeze(0)  # Shape: [1, 106]
            plddt_descriptors_sym_pos = torch.stack((plddt_1_sym_pos.expand(-1, plddt_2_sym_pos.size(1)),
                                                    plddt_2_sym_pos.expand(plddt_1_sym_pos.size(0), -1)), dim=-1)
        # Filter symmetry predictions
        pos_preds2 = pos_desc_dists2[pos_xyz_dists.T < 1.0]
        pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)
        pos_labels = torch.ones_like(pos_preds)
        pos_preds_p2 = pos_preds2
        pos_labels_p2 = pos_labels

        if flexibility:
            plddt_descriptors_sym_positive = plddt_descriptors_sym_pos[pos_xyz_dists.T < 1.0]

            # Concatenate positive descriptors
            plddt_pos_tot = torch.cat([plddt_descriptors_positive, plddt_descriptors_sym_positive], dim=0)

        # Symmetry for negative samples
        sample_desc1_2_id = torch.randperm(len(P1["embedding_2"]))[:n_desc_sample]
        sample_desc1_2 = P1["embedding_2"][sample_desc1_2_id]
        neg_preds_2 = torch.matmul(pos_descs2_2, sample_desc1_2.T).view(-1)
        neg_preds = torch.cat([neg_preds, neg_preds_2], dim=0)
        neg_labels = torch.zeros_like(neg_preds)
        neg_preds_p2 = neg_preds_2
        neg_labels_p2 = torch.zeros_like(neg_preds_2)

        if flexibility:
            # Get pLDDT values for negative symmetry
            plddt_1_sym_neg = P2["input_features"][P2["labels"] == 1][:, 16].unsqueeze(1)  # Shape: [99, 1]
            plddt_2_sym_neg = P1["input_features"][sample_desc1_2_id][:, 16].unsqueeze(0)  # Shape: [1, 100]
            plddt_descriptors_sym_neg = torch.stack((plddt_1_sym_neg.expand(-1, plddt_2_sym_neg.size(1)),
                                                    plddt_2_sym_neg.expand(plddt_1_sym_neg.size(0), -1)), dim=-1)
            plddt_descriptors_sym_neg = plddt_descriptors_sym_neg.view(-1, 2)

            # Concatenate negative descriptors
            plddt_neg_tot = torch.cat([plddt_descriptors_neg, plddt_descriptors_sym_neg], dim=0)

        n_points_sample = len(pos_labels)
        pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
        neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

        n_points_sample_p1_p2 = min(len(pos_preds_p1), len(pos_preds_p2))
        pos_indices_p1_p2 = torch.randperm(len(pos_labels_p1))[:n_points_sample_p1_p2]
        neg_indices_p1_sample = torch.randperm(len(neg_labels_p1))[:n_points_sample_p1_p2]
        neg_indices_p2_sample = torch.randperm(len(neg_labels_p2))[:n_points_sample_p1_p2]


        pos_preds = pos_preds[pos_indices]
        pos_labels = pos_labels[pos_indices]
        if flexibility:
            plddt_pos_tot = plddt_pos_tot[pos_indices]
        neg_preds = neg_preds[neg_indices]
        neg_labels = neg_labels[neg_indices]
        if flexibility:
            plddt_neg_tot = plddt_neg_tot[neg_indices]

        pos_preds_p1 = pos_preds_p1[pos_indices_p1_p2]
        pos_labels_p1 = pos_labels_p1[pos_indices_p1_p2]
        pos_preds_p2 = pos_preds_p2[pos_indices_p1_p2]
        pos_labels_p2 = pos_labels_p2[pos_indices_p1_p2]
        neg_preds_p1 = neg_preds_p1[neg_indices_p1_sample]
        neg_labels_p1 = neg_labels_p1[neg_indices_p1_sample]
        neg_preds_p2 = neg_preds_p2[neg_indices_p2_sample]
        neg_labels_p2 = neg_labels_p2[neg_indices_p2_sample]


        preds_concat = torch.cat([pos_preds, neg_preds])
        labels_concat = torch.cat([pos_labels, neg_labels])
        prob_preds = torch.sigmoid(preds_concat).tolist()

        pos_preds_p1_prob = torch.sigmoid(pos_preds_p1).tolist()
        pos_preds_p2_prob = torch.sigmoid(pos_preds_p2).tolist()
        neg_preds_p1_prob = torch.sigmoid(neg_preds_p1).tolist()
        neg_preds_p2_prob = torch.sigmoid(neg_preds_p2).tolist()

        preds_p1_final = pos_preds_p1_prob
        preds_p1_final.extend(neg_preds_p1_prob)
        preds_p2_final = pos_preds_p2_prob
        preds_p2_final.extend(neg_preds_p2_prob)

        results = {'P1':{'pred': preds_p1_final , 'labels':torch.cat([pos_labels_p1, neg_labels_p1])},\
                'P2': {'pred': preds_p2_final , 'labels': torch.cat([pos_labels_p2, neg_labels_p2])}}


        if flexibility:
            plddt_contact = torch.cat([plddt_pos_tot, plddt_neg_tot])
            plddt_contact = plddt_contact.tolist()
            #print(plddt_contact)
        
        loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)


    else:
        pos_preds = P1["iface_preds"][P1["labels"] == 1]
        pos_labels = P1["labels"][P1["labels"] == 1]
        neg_preds = P1["iface_preds"][P1["labels"] == 0]
        neg_labels = P1["labels"][P1["labels"] == 0]
        print(f'Positive: {(P1["labels"] == 1).sum() }')
        print(f'Negative: {(P1["labels"] == 0).sum() }')



        #n_points_sample = len(pos_labels)
        #pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
        #neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

        #pos_preds = pos_preds[pos_indices]
        #pos_labels = pos_labels[pos_indices]
        #neg_preds = neg_preds[neg_indices]
        #neg_labels = neg_labels[neg_indices]

        preds_concat = torch.cat([pos_preds, neg_preds])
        labels_concat = torch.cat([pos_labels, neg_labels])
        prob_preds = torch.sigmoid(preds_concat).tolist()
        bin_preds =  [1 if prob_preds[i] >= 0.5 else 0  for i in range(len(prob_preds))]
        print(f'Positive pred: {sum(bin_preds)}')
        print(f'Negative pred: {len(bin_preds) - sum(bin_preds)}')

        if flexibility:
            plddt_pos = P1["input_features"][P1["labels"].squeeze(dim=1) == 1][:, 16]# Shape: [99, 1]
            plddt_neg= P1["input_features"][P1["labels"].squeeze(dim=1) == 0][:, 16] # Shape: [1, 106]
            plddt_contact = torch.cat([plddt_pos, plddt_neg])
            plddt_contact = plddt_contact.tolist()
            loss = criterion(torch.sigmoid(preds_concat), labels_concat)
        else:
            loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
        results = {'P1':{'pred': preds_concat , 'labels':labels_concat}}
        

    if flexibility:
        return loss, preds_concat, labels_concat, plddt_contact, prob_preds, results
    else:
        return loss, preds_concat, labels_concat, prob_preds, results

def extract_single(P_batch, number, flex=False):
    P = {}  # First and second proteins
    batch = P_batch["batch"] == number
    batch_atoms = P_batch["batch_atoms"] == number

    with_mesh = P_batch["labels"] is not None
    # Ground truth labels are available on mesh vertices:
    P["labels"] = P_batch["labels"][batch] if with_mesh else None

    P["batch"] = P_batch["batch"][batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][batch]
    P["normals"] = P_batch["normals"][batch]

    # Atom information:
    P["atoms"] = P_batch["atoms"][batch_atoms]
    P["batch_atoms"] = P_batch["batch_atoms"][batch_atoms]

    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = P_batch["atom_xyz"][batch_atoms]
    P["atomtypes"] = P_batch["atomtypes"][batch_atoms]
    if flex:
        P["atomflex"] = P_batch["atomflex"][batch_atoms]

    return P


def save_protein_embeddings(pdb_id, embedding, labels, flex = None, bin_pred = None, save_dir="dmasif/predictions_site"):
    """Saves embeddings, labels, and flexibility for a single protein as .npy files, appending if the files exist."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Define paths for embedding, label, and flexibility files
    embedding_path = os.path.join(save_dir, f"{pdb_id}_emb.npy")
    label_path = os.path.join(save_dir, f"{pdb_id}_labels.npy")
    if flex is not None:
        flex_path = os.path.join(save_dir, f"{pdb_id}_flex.npy")
    if bin_pred is not None:
        pred_path = os.path.join(save_dir, f"{pdb_id}_pred.npy")

    # Check if embedding, label, and flexibility files exist
    
    new_embeddings = embedding.cpu().numpy()
    new_labels = labels.cpu().numpy()
    if flex is not None:
        new_flex = flex.cpu().numpy()
    if bin_pred is not None: 
        new_pred = torch.sigmoid(bin_pred).cpu().numpy()
        new_pred = (new_pred >= 0.5).astype(float)
    
    # Save embeddings, labels, and flexibility
    np.save(embedding_path, new_embeddings)
    np.save(label_path, new_labels)
    if flex is not None:
        np.save(flex_path, new_flex)
    if bin_pred is not None:
        np.save(pred_path, new_pred)

    print(f"Embeddings, labels, and flexibility for {pdb_id} saved successfully.")

def iterate(
    net,
    dataset,
    optimizer,
    args,
    warmup_scheduler=None,
    scheduler=None,
    test=False,
    save_path=None,
    pdb_ids=None,
    summary_writer=None,
    epoch_number=None,
    flex = False,
    criterion = None,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    total_processed_pairs = 0
    plddt_eval = {'plddt':[], 'probability':[], 'gt':[]}
    #prediction_aucroc = {'pdbid':[], 'AUC-ROC':[]}
    # Loop over one epoch:
    for it, protein_pair in enumerate(
        tqdm(dataset)
    ):  # , desc="Test " if test else "Train")):
        protein_batch_size = protein_pair.atom_coords_p1_batch[-1].item() + 1
        if save_path is not None:
            batch_ids = pdb_ids[
                total_processed_pairs : total_processed_pairs + protein_batch_size
            ]
            total_processed_pairs += protein_batch_size


        protein_pair.to(args.device)

        if not test:
            optimizer.zero_grad()

        # Generate the surface:
        torch.cuda.synchronize()
        surface_time = time.time()
        try:
            P1_batch, P2_batch = process(args, protein_pair, net, flex)
            #path_xyz = f"/disk1/flexibility/dmasif/predictions/{'_'.join(batch_ids[0].split('_')[:-1])}_xyz.npy"
            #np.save(path_xyz, P1_batch['xyz'].cpu().numpy())
        except:
            print('Something went wrong with the atomic representation')
            continue
        torch.cuda.synchronize()
        surface_time = time.time() - surface_time
        for protein_it in range(protein_batch_size):
            torch.cuda.synchronize()
            iteration_time = time.time()

            try:
                P1 = extract_single(P1_batch, protein_it, flex)
                P2 = None if args.single_protein else extract_single(P2_batch, protein_it, flex)
            except:
                continue


            if args.random_rotation:
                P1["rand_rot"] = protein_pair.rand_rot1.view(-1, 3, 3)[0]
                P1["atom_center"] = protein_pair.atom_center1.view(-1, 1, 3)[0]
                P1["xyz"] = P1["xyz"] - P1["atom_center"]
                P1["xyz"] = (
                    torch.matmul(P1["rand_rot"], P1["xyz"].T).T
                ).contiguous()
                P1["normals"] = (
                    torch.matmul(P1["rand_rot"], P1["normals"].T).T
                ).contiguous()
                if not args.single_protein:
                    P2["rand_rot"] = protein_pair.rand_rot2.view(-1, 3, 3)[0]
                    P2["atom_center"] = protein_pair.atom_center2.view(-1, 1, 3)[0]
                    P2["xyz"] = P2["xyz"] - P2["atom_center"]
                    P2["xyz"] = (
                        torch.matmul(P2["rand_rot"], P2["xyz"].T).T
                    ).contiguous()
                    P2["normals"] = (
                        torch.matmul(P2["rand_rot"], P2["normals"].T).T
                    ).contiguous()
            else:
                P1["rand_rot"] = torch.eye(3, device=P1["xyz"].device)
                P1["atom_center"] = torch.zeros((1, 3), device=P1["xyz"].device)
                if not args.single_protein:
                    P2["rand_rot"] = torch.eye(3, device=P2["xyz"].device)
                    P2["atom_center"] = torch.zeros((1, 3), device=P2["xyz"].device)
                    
            torch.cuda.synchronize()
            prediction_time = time.time()
            outputs = net(P1, P2)
            torch.cuda.synchronize()
            prediction_time = time.time() - prediction_time

            P1 = outputs["P1"]
            P2 = outputs["P2"]

            if args.search:
                generate_matchinglabels(args, P1, P2)

            if P1["labels"] is not None:
                if args.flexibility:
                    loss, sampled_preds, sampled_labels, plddt_contact, prob_preds, results_dict = compute_loss(args, P1, P2, all = True)
                else:
                    loss, sampled_preds, sampled_labels, prob_preds, results_dict = compute_loss(args, P1, P2, all = True)
            else:
                loss = torch.tensor(0.0)
                sampled_preds = None
                sampled_labels = None
            
            try:
                plddt_eval['plddt'].extend(plddt_contact)
                plddt_eval['probability'].extend(prob_preds)
                plddt_eval['gt'].extend(sampled_labels.cpu().tolist())
            except:
                len_list = len(prob_preds)
                plddt_eval['plddt'].extend([0]*len_list)
                plddt_eval['probability'].extend(prob_preds)
                plddt_eval['gt'].extend(sampled_labels.cpu().tolist())
            

            # Compute the gradient, update the model weights:
            if not test:
                torch.cuda.synchronize()
                back_time = time.time()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                torch.cuda.synchronize()
                back_time = time.time() - back_time
            

            if it == protein_it == 0 and not test:
                for para_it, (name, parameter) in enumerate(net.atomnet.named_parameters()):
                    if parameter.requires_grad:
                        #print(name, parameter)
                        summary_writer.add_histogram(
                            f"Gradients/Atomnet/para_{para_it}_{parameter.shape}",
                            parameter.grad.view(-1),
                            epoch_number,
                        )
                for para_it, parameter in enumerate(net.conv.parameters()):
                    if parameter.requires_grad:
                        summary_writer.add_histogram(
                            f"Gradients/Conv/para_{para_it}_{parameter.shape}",
                            parameter.grad.view(-1),
                            epoch_number,
                        )

                for d, features in enumerate(P1["input_features"].T):
                    summary_writer.add_histogram(f"Input features/{d}", features)

            """if save_path is not None:
                save_protein_batch_single(
                    batch_ids[protein_it], P1, save_path, pdb_idx=1
                )
                if not args.single_protein:
                    save_protein_batch_single(
                        batch_ids[protein_it], P2, save_path, pdb_idx=2
                    )"""

            try:
                if sampled_labels is not None:
                    predictions = prob_preds
                    binary_labels = (np.array(predictions) >= 0.5).astype(float)
                    roc_auc = roc_auc_score(np.rint(numpy(sampled_labels.view(-1))),numpy(sampled_preds.view(-1)),)
                    accuracy = accuracy_score(np.rint(numpy(sampled_labels.view(-1))), binary_labels,)
                    mcc = matthews_corrcoef( np.rint(numpy(sampled_labels.view(-1))), binary_labels,)
                    pr_auc = average_precision_score(np.rint(numpy(sampled_labels.view(-1))),numpy(sampled_preds.view(-1)),)
                    print('\n')
                    if args.search:
                        print(f'AUC-ROC: {roc_auc}')
                    else:
                        print(f'MCC: {mcc} | AUC-ROC: {roc_auc} | AUC-PR: {pr_auc}')
            except Exception as e:
                print("Problem with computing roc-auc")
                roc_auc = 0.0
                print(e)
                #continue

            R_values = outputs["R_values"]

        
            if args.search:
                info.append(
                    dict(
                        {
                            "Loss": loss.item(),
                            "ROC-AUC": roc_auc,
                            "conv_time": outputs["conv_time"],
                            "memory_usage": outputs["memory_usage"],
                        },
                        # Merge the "R_values" dict into "info", with a prefix:
                        **{"R_values/" + k: v for k, v in R_values.items()},
                        )
                    )
            else:
                info.append(
                    dict(
                        {
                            "Loss": loss.item(),
                            "ROC-AUC": roc_auc,
                            "PR-AUC": pr_auc,
                            "Accuracy": accuracy,
                            "MCC": mcc,
                            "conv_time": outputs["conv_time"],
                            "memory_usage": outputs["memory_usage"],
                        },
                        # Merge the "R_values" dict into "info", with a prefix:
                        **{"R_values/" + k: v for k, v in R_values.items()},
                    )
                )
            torch.cuda.synchronize()
            iteration_time = time.time() - iteration_time

            #path_pred = f"/disk1/flexibility/dmasif/predictions/{'_'.join(batch_ids[0].split('_')[:-1])}_pred.npy"
            #binary = torch.sigmoid(P1['iface_preds'])
            #binarized_tensor = (binary > 0.5).float()
            #np.save(path_pred, binarized_tensor.cpu().numpy())
            #path_pred = f"/disk1/flexibility/dmasif/predictions/{'_'.join(batch_ids[0].split('_')[:-1])}_labels.npy"
            #np.save(path_pred, P1["labels"].cpu().numpy())

            if args.save_emb:
                pdb_id, chain1, chain2 = batch_ids[0].split("_")
                pdbid_chain1 = f"{pdb_id}_{chain1}"
                pdbid_chain2 = f"{pdb_id}_{chain2}"

                if args.site:
                    pdbid_chain1 = f"{pdb_id}_{chain1}"
                    if args.flexibility:
                        save_protein_embeddings(pdbid_chain1, P1["embedding_1"], P1["labels"],  P1["input_features"][:, 16], P1["iface_preds"])
                    else:
                        save_protein_embeddings(pdbid_chain1, P1["embedding_1"], P1["labels"], bin_pred = P1["iface_preds"])
                    
                else:
                    if args.flexibility:
                        # Save the embeddings for each protein separately
                        save_protein_embeddings(pdbid_chain1, P1["embedding_1"], P1["labels"], P1["input_features"][:, 16])
                        save_protein_embeddings(pdbid_chain2, P2["embedding_2"], P2["labels"], P2["input_features"][:, 16])
                    else:
                        save_protein_embeddings(pdbid_chain1, P1["embedding_1"], P1["labels"])
                        save_protein_embeddings(pdbid_chain2, P2["embedding_2"], P2["labels"])

    # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)
    info = newdict

    results_df = pd.DataFrame(plddt_eval)

    # Save the DataFrame to a CSV file
    results_df.to_csv('protein_analysis_results.csv', index=False)

    # Final post-processing:
    return info

def iterate_surface_precompute(dataset, net, args):
    processed_dataset = []
    for it, protein_pair in enumerate(tqdm(dataset)):
        protein_pair.to(args.device)
        P1, P2 = process(args, protein_pair, net, args.flexibility)
        if args.random_rotation:
            P1["rand_rot"] = protein_pair.rand_rot1
            P1["atom_center"] = protein_pair.atom_center1
            P1["xyz"] = (
                torch.matmul(P1["rand_rot"].T, P1["xyz"].T).T + P1["atom_center"]
            )
            P1["normals"] = torch.matmul(P1["rand_rot"].T, P1["normals"].T).T
            if not args.single_protein:
                P2["rand_rot"] = protein_pair.rand_rot2
                P2["atom_center"] = protein_pair.atom_center2
                P2["xyz"] = (
                    torch.matmul(P2["rand_rot"].T, P2["xyz"].T).T + P2["atom_center"]
                )
                P2["normals"] = torch.matmul(P2["rand_rot"].T, P2["normals"].T).T
        protein_pair = protein_pair.to_data_list()[0]
        protein_pair.gen_xyz_p1 = P1["xyz"]
        protein_pair.gen_normals_p1 = P1["normals"]
        protein_pair.gen_batch_p1 = P1["batch"]
        protein_pair.gen_labels_p1 = P1["labels"]
        protein_pair.gen_xyz_p2 = P2["xyz"]
        protein_pair.gen_normals_p2 = P2["normals"]
        protein_pair.gen_batch_p2 = P2["batch"]
        protein_pair.gen_labels_p2 = P2["labels"]
        processed_dataset.append(protein_pair.to("cpu"))
    return processed_dataset