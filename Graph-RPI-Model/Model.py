import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import Linear, GINConv, GATConv
from torch_geometric.utils import degree, to_undirected
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred):
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss



class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # 第一层：使用 GATConv
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        # 第二层：使用 GINConv
        self.convs.append(GINConv(Linear(hidden_channels, out_channels), train_eps=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ELU()

    def forward(self, x, edge_index):
        edge_index = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0)))
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
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


class RPI(nn.Module):
    def __init__(self, encoder, edge_decoder, degree_decoder, mask):
        super(RPI, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.loss_fn = ce_loss

    def train_epoch(self, data, optimizer, alpha, batch_size=8192, grad_norm=1.0):
        x, edge_index = data.x, data.edge_index
        remaining_edges, masked_edges = self.mask(edge_index)
        num_neg = masked_edges.size(1)
        num_rna = data.num_rna
        neg_edges = torch.stack([
            torch.randint(0, num_rna, (num_neg,), device=edge_index.device),
            torch.randint(num_rna, data.num_nodes, (num_neg,), device=edge_index.device)
        ], dim=0)
        for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            z = self.encoder(x, remaining_edges)
            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            pos_out = self.edge_decoder(z, batch_masked_edges)
            neg_out = self.edge_decoder(z, batch_neg_edges)
            loss = self.loss_fn(pos_out, neg_out)
            deg = degree(masked_edges[1].flatten(), data.num_nodes).float()
            loss += alpha * F.mse_loss(self.degree_decoder(z).squeeze(), deg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        temp = torch.tensor(pred)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp.cpu())
        return auc, ap, acc, sen, pre, spe, F1, mcc



def mask_edge(edge_index, p):
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]


class Mask(nn.Module):
    def __init__(self, p):
        super(Mask, self).__init__()
        self.p = p

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges


def filter_bipartite_negatives(neg_edge_index, num_rna):
    mask = ((neg_edge_index[0] < num_rna) & (neg_edge_index[1] >= num_rna)) | \
           ((neg_edge_index[0] >= num_rna) & (neg_edge_index[1] < num_rna))
    return neg_edge_index[:, mask]



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