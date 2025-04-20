
import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from torch_sparse import SparseTensor
from captum.attr import KernelShap
from torch_geometric.nn import GATConv, GINConv, Linear

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2023)


sns.set_style("white")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.0




def read_protein_sequences_from_fasta(file_path):
    sequences = []
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences

def generate_features_protein_bert(sequences, tokenizer, transformer_model):
    features = []
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors='pt')["input_ids"]
        with torch.no_grad():
            hidden_states = transformer_model(inputs)[0]
        embedding_mean = torch.mean(hidden_states[0], dim=0)
        features.append(embedding_mean)
    features_np = np.vstack([feat.numpy() for feat in features])
    print(f"Generated protein BERT features with shape: {features_np.shape}")
    return features_np

def parse_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                ids.append(line[1:].strip())
    return ids

def parse_pairs(file_path):
    positive_pairs = []
    negative_pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')
            if len(fields) == 2:
                protein_id, rna_id = fields
                positive_pairs.append((protein_id, rna_id))
            elif len(fields) >= 3:
                protein_id, rna_id, label = fields[:3]
                if int(label) == 1:
                    positive_pairs.append((protein_id, rna_id))
                elif int(label) == 0:
                    negative_pairs.append((protein_id, rna_id))
    return positive_pairs, negative_pairs

def parse_pairs_only(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                protein_id, rna_id = fields[:2]
                pairs.append((protein_id, rna_id))
    return pairs


def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(pos_out))
    return pos_loss + neg_loss

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GINConv(Linear(hidden_channels, out_channels), train_eps=True))
        self.bns.append(nn.BatchNorm1d(out_channels))
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, x, edge_index):
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0)))
        x = self.dropout(x)
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[1](x, edge_index)
        x = self.bns[1](x)
        x = self.activation(x)
        return x

class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(EdgeDecoder, self).__init__()
        self.mlps = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, out_channels)
        ])
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        for mlp in self.mlps[:-1]:
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        return x

class DegreeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(DegreeDecoder, self).__init__()
        self.mlps = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, out_channels)
        ])
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, x):
        for mlp in self.mlps[:-1]:
            x = mlp(x)
            x = self.dropout(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        x = self.activation(x)
        return x

class Mask(nn.Module):
    def __init__(self, p):
        super(Mask, self).__init__()
        self.p = p

    def forward(self, edge_index):
        return edge_index, edge_index

class RPI(nn.Module):
    def __init__(self, encoder, edge_decoder, degree_decoder, mask):
        super(RPI, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.loss_fn = ce_loss

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return z

def forward_edge_score(model, x_uv, node_u, node_v, edge_index, full_x):
    x_modified = full_x.clone()
    x_modified[node_u] = x_uv[0]
    x_modified[node_v] = x_uv[1]
    z = model.encoder(x_modified, edge_index)
    edge_tensor = torch.tensor([[node_u], [node_v]], dtype=torch.long, device=full_x.device)
    score = model.edge_decoder(z, edge_tensor).squeeze()
    return score

def shap_explain_uv(model, full_x, edge_index, node_u, node_v, baseline="zeros", nsamples=200):

    x_u = full_x[node_u].clone().unsqueeze(0)
    x_v = full_x[node_v].clone().unsqueeze(0)
    x_uv = torch.cat([x_u, x_v], dim=0)
    if baseline == "zeros":
        baseline_x_uv = torch.zeros_like(x_uv)
    elif baseline == "mean":
        baseline_x_uv = x_uv.mean(dim=0, keepdim=True).repeat(2, 1)
    else:
        raise ValueError("Unsupported baseline mode")

    def forward_callable(x_uv_input):
        outputs = []
        for i in range(x_uv_input.size(0)):
            score = forward_edge_score(
                model=model,
                x_uv=x_uv_input[i],
                node_u=node_u,
                node_v=node_v,
                edge_index=edge_index,
                full_x=full_x
            )
            outputs.append(score)
        return torch.stack(outputs, dim=0)

    ks = KernelShap(forward_callable)
    inputs_for_shap = x_uv.unsqueeze(0)
    baseline_for_shap = baseline_x_uv.unsqueeze(0)
    shap_attr = ks.attribute(inputs=inputs_for_shap,
                             baselines=baseline_for_shap,
                             n_samples=nsamples)
    shap_values_uv = shap_attr[0]
    return shap_values_uv


def visualize_shap_feature_groups(shap_values, feature_group_dict, node_type="RNA", save_path=None):
    # group-level mean |SHAP|
    group_names = []
    mean_importances = []
    for key, (start, end) in feature_group_dict.items():
        group_names.append(key)
        group_val = torch.mean(torch.abs(shap_values[start:end])).item()
        mean_importances.append(group_val)

    plt.figure(figsize=(8, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor("white")

    max_val = max(mean_importances)
    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    colors = [cmap(v / max_val) for v in mean_importances]

    bars = plt.bar(group_names, mean_importances, color=colors)

    plt.grid(False)

    plt.xlabel(f"{node_type} Feature Group", fontsize=12, color='black')
    plt.ylabel("Mean |SHAP| Value", fontsize=12, color='black')
    plt.title(f"{node_type} Node Feature Group Importance", fontsize=14, color='black')
    plt.xticks(rotation=45, color='black')
    plt.yticks(color='black')


    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='black')
        plt.close()
        print(f"{node_type} feature group chart saved to {save_path}")
    else:
        plt.show()

def visualize_top_features_with_names(shap_vector, feature_names, top_k=20, node_type="RNA", save_path=None):
    shap_np = np.abs(shap_vector.cpu().numpy())
    sorted_indices = np.argsort(shap_np)[::-1]
    top_indices = sorted_indices[:top_k]
    top_values = shap_np[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(8, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor("white")

    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    max_shap = top_values.max()
    colors = [cmap(val / max_shap) for val in top_values]

    plt.barh(top_features, top_values, color=colors)
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.xlabel("Mean |SHAP| Value", fontsize=12, color='black')
    plt.title(f"Top {top_k} {node_type} Features SHAP Importance", fontsize=14, color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')


    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='black')
        plt.close()
        print(f"Top {top_k} {node_type} feature chart saved to {save_path}")
    else:
        plt.show()




def load_rna_feature_names(rna_csv_path):
    df = pd.read_csv(rna_csv_path)
    return list(df.columns)

def load_protein_feature_names(protein_csv_path):
    df = pd.read_csv(protein_csv_path)
    artificial_names = list(df.columns)  # assume length=589
    esm_names = [f"ESM_{i}" for i in range(1, 321)]
    return artificial_names + esm_names  # total=909

def select_pair_from_train(pair_file, target_protein, target_rna):
    with open(pair_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                protein_id, rna_id = parts[0], parts[1]
                if protein_id == target_protein and rna_id == target_rna:
                    return protein_id, rna_id
    return None


rna_feature_groups = {
    "NAC": (0, 4),
    "Kmer": (4, 68),
    "DPCP": (68, 164),
    "PseDNC": (164, 183),
    "PCPseDNC": (183, 192),
    "CKSNAP": (192, 256)
}
protein_feature_groups = {
    "AAC": (0, 20),
    "PAAC": (20, 43),
    "CKSAAGP": (43, 143),
    "QSOrder": (143, 189),
    "DDE": (189, 589),
    "ESM-2": (589, 909)
}


if __name__ == "__main__":
    ds_save_path = "Fearture"
    output_dim = 909
    fold = 3
    print(f"\n--- Processing Fold {fold} ---")
    fold_dir = os.path.join(ds_save_path, f"fold{fold}")


    test_x = torch.load(os.path.join(fold_dir, "test", "x.pt"))
    test_edge_index = torch.load(os.path.join(fold_dir, "test", "edge_index.pt"))
    test_pos = torch.load(os.path.join(fold_dir, "test", "pos_edge_label_index.pt"))
    test_neg = torch.load(os.path.join(fold_dir, "test", "neg_edge_label_index.pt"))
    test_data = Data(x=test_x, edge_index=test_edge_index)
    test_data.pos_edge_label_index = test_pos
    test_data.neg_edge_label_index = test_neg

    encoder = GNNEncoder(in_channels=output_dim, hidden_channels=64, out_channels=128, heads=8)
    edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    mask = Mask(p=0.4)
    model = RPI(encoder, edge_decoder, degree_decoder, mask)

    best_model_path = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
    state_dict = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    protein_file = os.path.join(ds_save_path, "merge", "Protein.txt")
    rna_file = os.path.join(ds_save_path, "merge", "RNA.txt")
    protein_ids = parse_ids(protein_file)
    rna_ids = parse_ids(rna_file)
    protein_id_to_index = {pid: idx for idx, pid in enumerate(protein_ids)}
    rna_id_to_index = {rid: idx for idx, rid in enumerate(rna_ids)}

    pair_file = os.path.join(fold_dir, "train", "train_pairs.txt")
    target_protein = "1FFK-P"
    target_rna = "1FFK-0"
    selected_pair = select_pair_from_train(pair_file, target_protein, target_rna)
    if selected_pair is None:
        print("Target pair not found in train_pairs.txt.")
        exit(1)
    else:
        protein_id, rna_id = selected_pair
        node_u = rna_id_to_index[rna_id]
        node_v = protein_id_to_index[protein_id] + len(rna_ids)
        print(f"Selected pair: Protein {protein_id}, RNA {rna_id}")
        print(f"Corresponding node indices: RNA={node_u}, Protein={node_v}")

    rna_csv_path = "RNA_features.csv"
    protein_csv_path = "protein_features.csv"
    rna_feature_names = load_rna_feature_names(rna_csv_path)        # len=256
    protein_feature_names = load_protein_feature_names(protein_csv_path)  # len=909


    shap_vals = shap_explain_uv(
        model=model,
        full_x=test_data.x,
        edge_index=test_data.edge_index,
        node_u=node_u,
        node_v=node_v,
        baseline="zeros",
        nsamples=200
    )
    print(f"\nFold {fold}: SHAP for RNA node {node_u}, Protein node {node_v}")
    print(shap_vals)


    shap_rna = shap_vals[0, :256]

    shap_protein = shap_vals[1, :909]

    visualize_shap_feature_groups(
        shap_values=shap_rna,
        feature_group_dict=rna_feature_groups,
        node_type="RNA",
        save_path=os.path.join(fold_dir, "RNA_shap.png")
    )
    visualize_shap_feature_groups(
        shap_values=shap_protein,
        feature_group_dict=protein_feature_groups,
        node_type="Protein",
        save_path=os.path.join(fold_dir, "Protein_shap.png")
    )


    visualize_top_features_with_names(
        shap_vector=shap_rna,
        feature_names=rna_feature_names,
        top_k=20,
        node_type="RNA",
        save_path=os.path.join(fold_dir, "RNA_top20.png")
    )
    visualize_top_features_with_names(
        shap_vector=shap_protein,
        feature_names=protein_feature_names,
        top_k=20,
        node_type="Protein",
        save_path=os.path.join(fold_dir, "Protein_top20.png")
    )

    print(f"Fold {fold} SHAP explanation and visualization saved.")
    print("\n[Done] Fold 3 SHAP analysis and figures saved.")
