import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from pykeops.torch import LazyTensor
from src.data_iteration import process, extract_single 

from src.geometry_processing import (
    curvatures,
    mesh_normals_areas,
    tangent_vectors,
    atoms_to_points_normals,
)
from src.helper import soft_dimension, diagonal_ranges
from src.benchmark_models import DGCNN_seg, PointNet2_seg, dMaSIFConv_seg
import warnings
warnings.filterwarnings("ignore", message="torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats")

"""class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        return 100*(input > threshold).float()  # Forward pass: strict binary

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Backward pass: pass gradients as-is"""

class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return 100*(input > 80).float()  # Forward pass: strict binary

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Backward pass: pass gradients as-is



def knn_atoms(x, y, x_batch, y_batch, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def get_atom_features(x, y, x_batch, y_batch, y_atomtype, k=16):

    idx, dists = knn_atoms(x, y, x_batch, y_batch, k=k)  # (num_points, k)
    num_points, _ = idx.size()

    idx = idx.view(-1)
    dists = 1 / dists.view(-1, 1)
    _, num_dims = y_atomtype.size()

    feature = y_atomtype[idx, :]
    feature = torch.cat([feature, dists], dim=1)
    feature = feature.view(num_points, k, num_dims + 1)

    return feature


class Atom_embedding(nn.Module):
    def __init__(self, args):
        super(Atom_embedding, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        self.conv1 = nn.Linear(self.D + 1, self.D)
        self.conv2 = nn.Linear(self.D, self.D)
        self.conv3 = nn.Linear(2 * self.D, self.D)
        self.bn1 = nn.BatchNorm1d(self.D)
        self.bn2 = nn.BatchNorm1d(self.D)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        fx = get_atom_features(x, y, x_batch, y_batch, y_atomtypes, k=self.k)
        fx = self.conv1(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn1(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx1 = fx.sum(dim=1, keepdim=False)

        fx = self.conv2(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn2(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx2 = fx.sum(dim=1, keepdim=False)
        fx = torch.cat((fx1, fx2), dim=-1)
        fx = self.conv3(fx)

        return fx


class AtomNet(nn.Module):
    def __init__(self, args):
        super(AtomNet, self).__init__()
        self.args = args

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.embed = Atom_embedding(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        return self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)

class Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        #self.k = 17
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        
        k = self.k
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomtypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atomtypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb
    
class Atom_embedding_flex_MP(nn.Module):
    def __init__(self, args):
        super(Atom_embedding_flex_MP, self).__init__()
        self.D = 1
        self.k = 16
        self.args = args
        #self.k = 17
        """self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(1, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)"""

    def forward(self, x, y, y_atomflex, x_batch, y_batch, threshold):
        
        k = self.k
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomflex.shape[-1]
        features = y_atomflex[idx.reshape(-1), :]
        #features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
        #features = features.view(num_points, k, num_dims + 1)
        features = features.view(num_points, k, num_dims)
        features = features.mean(dim=1)
        if self.args.binary:
            #features = features * (features > 80).float()
            features = BinarySTE.apply(features)
            #print('Binary')

        return features

class Atom_Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 17
        #self.k = 18
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomtypes.shape[0]

        out = y_atomtypes
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out

class Atom_Atom_embedding_flex_MP(nn.Module):
    def __init__(self, args):
        super(Atom_Atom_embedding_flex_MP, self).__init__()
        self.D = 1
        self.k = 17
        #self.k = 18
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(1, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomtypes.shape[0]

        out = y_atomtypes
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out

class AtomNet_MP(nn.Module):
    def __init__(self, args):
        super(AtomNet_MP, self).__init__()
        self.args = args

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
        )

        self.embed = Atom_embedding_MP(args)
        self.atom_atom = Atom_Atom_embedding_MP(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch, atomflex=None):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        atomtypes = self.atom_atom(
            atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        )
        atomtypes = self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)

        return atomtypes

class AtomNet_MP_flex(nn.Module):
    def __init__(self, args):
        super(AtomNet_MP_flex, self).__init__()
        self.args = args
        self.threshold = args.threshold

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
        )
        self.embed = Atom_embedding_MP(args)
        self.atom_atom = Atom_Atom_embedding_MP(args)
        self.embed_flex = Atom_embedding_flex_MP(args)
        self.atom_atom_flex = Atom_Atom_embedding_flex_MP(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch, atomflex=None):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        atomtypes = self.atom_atom(
            atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        )
        atomtypes = self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)

        #atomflex = self.transform_types(atomflex)
        atomflex = self.atom_atom_flex(
            atom_xyz, atom_xyz, atomflex, atom_batch, atom_batch
        )
        atomflex = self.embed_flex(xyz, atom_xyz, atomflex, batch, atom_batch, self.threshold)
        return atomtypes, atomflex
    


def combine_pair(P1, P2):
    P1P2 = {}
    for key in P1:
        v1 = P1[key]
        v2 = P2[key]
        if v1 is None:
            continue

        if key == "batch" or key == "batch_atoms":
            v1v2 = torch.cat([v1, v2 + v1[-1] + 1], dim=0)
        elif key == "triangles":
            # v1v2 = torch.cat([v1,v2],dim=1)
            continue
        else:
            v1v2 = torch.cat([v1, v2], dim=0)
        P1P2[key] = v1v2

    return P1P2


def split_pair(P1P2):
    batch_size = P1P2["batch_atoms"][-1] + 1
    p1_indices = P1P2["batch"] < torch.div(batch_size, 2, rounding_mode='floor')
    p2_indices = P1P2["batch"] >= torch.div(batch_size, 2, rounding_mode='floor')

    p1_atom_indices = P1P2["batch_atoms"] < torch.div(batch_size, 2, rounding_mode='floor')
    p2_atom_indices = P1P2["batch_atoms"] >= torch.div(batch_size, 2, rounding_mode='floor')

    P1 = {}
    P2 = {}
    for key in P1P2:
        v1v2 = P1P2[key]

        if (key == "rand_rot") or (key == "atom_center"):
            n = torch.div(v1v2.shape[0], 2, rounding_mode='floor')
            P1[key] = v1v2[:n].view(-1, 3)
            P2[key] = v1v2[n:].view(-1, 3)
        elif "atom" in key:
            P1[key] = v1v2[p1_atom_indices]
            P2[key] = v1v2[p2_atom_indices]
        elif key == "triangles":
            continue
            # P1[key] = v1v2[:,p1_atom_indices]
            # P2[key] = v1v2[:,p2_atom_indices]
        else:
            P1[key] = v1v2[p1_indices]
            P2[key] = v1v2[p2_indices]

    P2["batch"] = P2["batch"] - batch_size + 1
    P2["batch_atoms"] = P2["batch_atoms"] - batch_size + 1

    return P1, P2



def project_iface_labels(P, threshold=2.0):

    queries = P["xyz"]
    batch_queries = P["batch"]
    source = P["mesh_xyz"]
    batch_source = P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1, 1) < threshold
    ).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein
    query_labels = labels[nn_i] * nn_dist_i
    P["labels"] = query_labels

class dMaSIF(nn.Module):
    def __init__(self, args):
        super(dMaSIF, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.curvature_scales = args.curvature_scales
        self.args = args

        I = args.in_channels
        O = args.orientation_units
        E = args.emb_dims
        H = args.post_units

        # Computes chemical features
        if args.flexibility:
            self.atomnet = AtomNet_MP_flex(args)
        else:
            self.atomnet = AtomNet_MP(args)
            print('No flexibility')
        self.dropout = nn.Dropout(args.dropout)

        if args.embedding_layer == "dMaSIF":
            # Post-processing, without batch norm:
            self.orientation_scores = nn.Sequential(
                nn.Linear(I, O),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(O, 1),
            )

            # Segmentation network:
            self.conv = dMaSIFConv_seg(
                args,
                in_channels=I,
                out_channels=E,
                n_layers=args.n_layers,
                radius=args.radius,
            )

            # Asymmetric embedding
            if args.search:
                self.orientation_scores2 = nn.Sequential(
                    nn.Linear(I, O),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(O, 1),
                )

                self.conv2 = dMaSIFConv_seg(
                    args,
                    in_channels=I,
                    out_channels=E,
                    n_layers=args.n_layers,
                    radius=args.radius,
                )

        elif args.embedding_layer == "DGCNN":
            self.conv = DGCNN_seg(I + 3, E,self.args.n_layers,self.args.k)
            if args.search:
                self.conv2 = DGCNN_seg(I + 3, E,self.args.n_layers,self.args.k)

        elif args.embedding_layer == "PointNet++":
            self.conv = PointNet2_seg(args, I, E)
            if args.search:
                self.conv2 = PointNet2_seg(args, I, E)

        if args.site:     
            self.net_out = nn.Sequential(
                nn.Linear(E, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, 1),
            )

    def features(self, P, i=1):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""
        if (
            not self.args.use_mesh and "xyz" not in P
        ):  # Compute the pseudo-surface directly from the atoms
            # (Note that we use the fact that dicts are "passed by reference" here)
            P["xyz"], P["normals"], P["batch"] = atoms_to_points_normals(
                P["atoms"],
                P["batch_atoms"],
                atomtypes=P["atomtypes"],
                resolution=self.args.resolution,
                sup_sampling=self.args.sup_sampling,
                atomflex = P["atomflex"] if self.args.flexibility else None,
            )

        # Estimate the curvatures using the triangles or the estimated normals:
        P_curvatures = curvatures(
            P["xyz"],
            triangles=P["triangles"] if self.args.use_mesh else None,
            normals=None if self.args.use_mesh else P["normals"],
            scales=self.curvature_scales,
            batch=P["batch"],
        )

        # Compute chemical features on-the-fly: #TODO: Control the flexibility embeddings because the flexibility changes to 1.0
        if self.args.flexibility:
            chemfeats, flexibility = self.atomnet(
                P["xyz"], P["atom_xyz"], P["atomtypes"], P["batch"], P["batch_atoms"], P["atomflex"],
            )
        else:
            chemfeats = self.atomnet(
                P["xyz"], P["atom_xyz"], P["atomtypes"], P["batch"], P["batch_atoms"],
            )

        if self.args.no_chem:
            chemfeats = 0.0 * chemfeats
        if self.args.no_geom:
            P_curvatures = 0.0 * P_curvatures
        if self.args.flexibility and self.args.no_flex:
            flexibility = 0.0 * flexibility

        # Concatenate our features:
        if self.args.flexibility:
            return torch.cat([P_curvatures, chemfeats, flexibility], dim=1).contiguous()
        else:
            return torch.cat([P_curvatures, chemfeats], dim=1).contiguous()

    def embed(self, P):
        """Embeds all points of a protein in a high-dimensional vector space."""

        features = self.features(P)
        if self.args.weight!=1:
            features[-1] = self.args.weight * features[-1]
        if self.args.recurrent:
            recurrent_flex = features[:, -1].unsqueeze(1)
        else:
            recurrent_flex = None
        features = self.dropout(features)
        P["input_features"] = features

        torch.cuda.synchronize(device=features.device)
        torch.cuda.reset_max_memory_allocated(device=P["atoms"].device)
        begin = time.time()

        # Ours:
        if self.args.embedding_layer == "dMaSIF":
            self.conv.load_mesh(
                P["xyz"],
                triangles=P["triangles"] if self.args.use_mesh else None,
                normals=None if self.args.use_mesh else P["normals"],
                weights=self.orientation_scores(features),
                batch=P["batch"],
            )
            P["embedding_1"] = self.conv(features, recurrent_flex)
            if self.args.search:
                self.conv2.load_mesh(
                    P["xyz"],
                    triangles=P["triangles"] if self.args.use_mesh else None,
                    normals=None if self.args.use_mesh else P["normals"],
                    weights=self.orientation_scores2(features),
                    batch=P["batch"],
                )
                P["embedding_2"] = self.conv2(features, recurrent_flex)

        # First baseline:
        elif self.args.embedding_layer == "DGCNN":
            features = torch.cat([features, P["xyz"]], dim=-1).contiguous()
            P["embedding_1"] = self.conv(P["xyz"], features, P["batch"])
            if self.args.search:
                P["embedding_2"] = self.conv2(
                    P["xyz"], features, P["batch"]
                )

        # Second baseline
        elif self.args.embedding_layer == "PointNet++":
            P["embedding_1"] = self.conv(P["xyz"], features, P["batch"])
            if self.args.search:
                P["embedding_2"] = self.conv2(P["xyz"], features, P["batch"])

        torch.cuda.synchronize(device=features.device)
        end = time.time()
        memory_usage = torch.cuda.max_memory_allocated(device=P["atoms"].device)
        conv_time = end - begin

        return conv_time, memory_usage

    def preprocess_surface(self, P, flex):
        P["xyz"], P["normals"], P["batch"] = atoms_to_points_normals(
            P["atoms"],
            P["batch_atoms"],
            atomtypes=P["atomtypes"],
            resolution=self.args.resolution,
            sup_sampling=self.args.sup_sampling,
            distance=self.args.distance,
            atomflex=P["atomflex"] if flex else None,
        )
        if P['mesh_labels'] is not None:
            project_iface_labels(P)

    def forward(self, P1, P2=None):
        # Compute embeddings of the point clouds:
        if P2 is not None:
            P1P2 = combine_pair(P1, P2)
        else:
            P1P2 = P1

        conv_time, memory_usage = self.embed(P1P2)

        # Monitor the approximate rank of our representations:
        R_values = {}
        R_values["input"] = soft_dimension(P1P2["input_features"])
        R_values["conv"] = soft_dimension(P1P2["embedding_1"])

        if self.args.site:
            P1P2["iface_preds"] = self.net_out(P1P2["embedding_1"])

        if P2 is not None:
            P1, P2 = split_pair(P1P2)
        else:
            P1 = P1P2

        return {
            "P1": P1,
            "P2": P2,
            "R_values": R_values,
            "conv_time": conv_time,
            "memory_usage": memory_usage,
        }

class ExpModel(nn.Module):
    def __init__(self, model_path, args):
        super(ExpModel, self).__init__()

        self.masif_model = dMaSIF(args)
        self.args = args
        # net.load_state_dict(torch.load(model_path, map_location=args.device))
        self.masif_model.load_state_dict(
            torch.load(model_path, map_location=args.device)["model_state_dict"]
        )
        self.masif_model.to(args.device)


    def __pre_process_data(self, protein_pair):
        print(protein_pair, type(protein_pair))
        protein_batch_size = protein_pair.atom_coords_p1_batch[-1].item() + 1
        protein_pair.to(self.args.device)
        
        P1_batch, P2_batch = process(self.args, protein_pair, self.masif_model, self.args.flexibility)

        for protein_it in range(protein_batch_size):
            torch.cuda.synchronize()

            P1 = extract_single(P1_batch, protein_it, self.args)
            P2 = None if self.args.single_protein else extract_single(P2_batch, protein_it, self.args)


            if self.args.random_rotation:
                P1["rand_rot"] = protein_pair.rand_rot1.view(-1, 3, 3)[0]
                P1["atom_center"] = protein_pair.atom_center1.view(-1, 1, 3)[0]
                P1["xyz"] = P1["xyz"] - P1["atom_center"]
                P1["xyz"] = (
                    torch.matmul(P1["rand_rot"], P1["xyz"].T).T
                ).contiguous()
                P1["normals"] = (
                    torch.matmul(P1["rand_rot"], P1["normals"].T).T
                ).contiguous()
                if not self.args.single_protein:
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
                if not self.args.single_protein:
                    P2["rand_rot"] = torch.eye(3, device=P2["xyz"].device)
                    P2["atom_center"] = torch.zeros((1, 3), device=P2["xyz"].device)
                    
            return P1, P2

    def forward(self, x):
        p1,p2 = self.__pre_process_data(x)
        print(type(p1))
        outputs = self.masif_model(p1,p2)
        P1 = outputs["P1"]["labels"]
        P2 = outputs["P2"]["labels"]

        out = torch.concat((P1,P2), 0)
        return out