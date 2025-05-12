

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
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd



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

        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))

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


# 假设你已有 RNA 与蛋白质其他特征生成函数
from predict_model.RNA_feature import generate_features_rna
from predict_model.protein_feature import generate_features_protein



def build_edge_index(pairs, protein_id_to_index, rna_id_to_index):
    edge_list = []
    for p, r in pairs:
        if p in protein_id_to_index and r in rna_id_to_index:
            rna_idx = rna_id_to_index[r]
            prot_idx = protein_id_to_index[p] + len(rna_id_to_index)
            edge_list.append([rna_idx, prot_idx])
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.LongTensor(edge_list).t()



def save_pairs(edge_label_index, file_path):
    with open(file_path, 'w') as f:
        for i in range(edge_label_index.size(1)):
            idx1 = edge_label_index[0, i].item()
            idx2 = edge_label_index[1, i].item()
            if idx1 < len(rna_ids) and idx2 >= len(rna_ids):
                rna_id = rna_ids[idx1]
                protein_id = protein_ids[idx2 - len(rna_ids)]
            elif idx2 < len(rna_ids) and idx1 >= len(rna_ids):
                rna_id = rna_ids[idx2]
                protein_id = protein_ids[idx1 - len(rna_ids)]
            else:
                continue
            f.write(f"{protein_id}\t{rna_id}\n")


def train_and_evaluate(model, train_data, test_data, optimizer, num_epochs=230, alpha=0.5):
    best_acc = 0
    best_state = None
    best_metrics = None
    for epoch in range(num_epochs):
        model.train()
        model.train_epoch(train_data, optimizer, alpha)
        model.eval()
        with torch.no_grad():
            z = model.encoder(test_data.x, test_data.edge_index)
            metrics = model.test(
                z,
                test_data.pos_edge_label_index,
                test_data.neg_edge_label_index
            )
        if metrics[2] > best_acc:
            best_acc = metrics[2]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:03d}: Test ACC={metrics[2]:.4f}")
    return best_metrics, best_state
def get_negative_samples(neg_file, data_all, num_rna, num_pos, device, strategy="file"):
    if strategy == "file":

        neg_pairs = parse_pairs_only(neg_file)
        neg_edge_index = build_edge_index(neg_pairs)
    elif strategy == "random":

        transform = T.RandomLinkSplit(
            num_val=0,
            num_test=0,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=True
        )
        temp_data, _, _ = transform(data_all)
        neg_edge_index = filter_bipartite_negatives(temp_data.neg_edge_label_index, num_rna)
    elif strategy == "mixed":

        neg_pairs = parse_pairs_only(neg_file)
        neg_edge_index_file = build_edge_index(neg_pairs)
        transform = T.RandomLinkSplit(
            num_val=0,
            num_test=0,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=True
        )
        temp_data, _, _ = transform(data_all)
        neg_edge_index_rand = filter_bipartite_negatives(temp_data.neg_edge_label_index, num_rna)
        neg_edge_index = torch.cat([neg_edge_index_file, neg_edge_index_rand], dim=1)
    elif strategy == "hard":

        transform = T.RandomLinkSplit(
            num_val=0,
            num_test=0,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=True
        )
        temp_data, _, _ = transform(data_all)
        neg_edge_index = filter_bipartite_negatives(temp_data.neg_edge_label_index, num_rna)
    else:
        raise ValueError("Unknown negative sampling strategy!")


    if neg_edge_index.size(1) < num_pos:
        indices = torch.randint(0, neg_edge_index.size(1), (num_pos,), device=device)
        neg_edge_index = neg_edge_index[:, indices]
    elif neg_edge_index.size(1) > num_pos:
        indices = torch.randperm(neg_edge_index.size(1))[:num_pos]
        neg_edge_index = neg_edge_index[:, indices]

    return neg_edge_index

if __name__ == '__main__':

    dataset_name = "RPImerged"
    base_path = "/Users/ggcl7/Desktop/RPI审稿意见回复511/消融参数mask"
    data_path = os.path.join(base_path, dataset_name)
    folds_path = os.path.join(data_path, "folds")
    save_path = os.path.join(base_path, dataset_name)
    os.makedirs(save_path, exist_ok=True)


    protein_file = os.path.join(data_path, "Protein.txt")
    rna_file = os.path.join(data_path, "RNA.txt")
    pairs_file = os.path.join(data_path, "Pairs.txt")

    protein_sequences = read_protein_sequences_from_fasta(protein_file)

    rna_features = generate_features_rna(rna_file)

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("ESM_Pre_model", trust_remote_code=True)
    transformer_model = AutoModel.from_pretrained("ESM_Pre_model", trust_remote_code=True)
    protein_bert_features = generate_features_protein_bert(protein_sequences, tokenizer, transformer_model)

    protein_new_features = generate_features_protein(protein_file)
    print(f"Generated protein new features with shape: {protein_new_features.shape}")

    output_dim = 909
    RNA = rna_features
    protein = np.hstack((protein_bert_features, protein_new_features))

    protein_ids = parse_ids(protein_file)
    rna_ids = parse_ids(rna_file)
    protein_id_to_index = {pid: idx for idx, pid in enumerate(protein_ids)}
    rna_id_to_index = {rid: idx for idx, rid in enumerate(rna_ids)}

    rna_emb = []
    for feat in RNA:
        feat_list = feat.tolist()
        feat_list += [0] * (output_dim - len(feat_list))
        rna_emb.append(feat_list)
    rna_emb = torch.Tensor(rna_emb)

    protein_emb = []
    for feat in protein:
        feat_list = feat.tolist()
        feat_list += [0] * (output_dim - len(feat_list))
        protein_emb.append(feat_list)
    protein_emb = torch.Tensor(protein_emb)

    feature = torch.cat([rna_emb, protein_emb], dim=0)

    full_positive_pairs, _ = parse_pairs(pairs_file)
    relation_matrix = np.zeros((len(rna_ids), len(protein_ids)), dtype=int)
    for protein_id, rna_id in full_positive_pairs:
        if (protein_id in protein_id_to_index) and (rna_id in rna_id_to_index):
            protein_index = protein_id_to_index[protein_id]
            rna_index = rna_id_to_index[rna_id]
            relation_matrix[rna_index, protein_index] = 1
    pos_edge_index = []
    for rna_index in range(len(rna_ids)):
        for protein_index in range(len(protein_ids)):
            if relation_matrix[rna_index, protein_index] == 1:
                pos_edge_index.append([rna_index, protein_index + len(rna_ids)])
    pos_edge_index = torch.LongTensor(pos_edge_index).t()

    data_all = Data(x=feature, edge_index=pos_edge_index)
    data_all.num_rna = len(rna_ids)

    conv_combos = [("GATConv", "GINConv")]

    neg_sampling_strategies = ["none"]

    all_results = []
    save_path_conv = os.path.join(save_path, "negative_sampling")
    os.makedirs(save_path_conv, exist_ok=True)

    for conv_combo in conv_combos:
        for neg_strategy in neg_sampling_strategies:
            print(f"\n=== Testing Conv: {conv_combo[0]}+{conv_combo[1]}, Negative Sampling: {neg_strategy} ===")
            combo_results = []

            for fold in range(1, 6):
                print(f"\n🔁 Fold {fold}")
                set_seed(2023 + fold)


                test_pos, _ = parse_pairs(os.path.join(folds_path, f"fold{fold}_pairs.txt"))
                train_pos = []
                for i in range(1,6):
                    if i!=fold:
                        tp, _ = parse_pairs(os.path.join(folds_path, f"fold{i}_pairs.txt"))
                        train_pos.extend(tp)


                transform = T.RandomLinkSplit(
                    num_val=0,
                    num_test=0.2,
                    is_undirected=True,
                    split_labels=True,
                    add_negative_train_samples=True
                )
                train_data, _, test_data = transform(data_all)

                pid2idx = {pid:i for i,pid in enumerate(protein_ids)}
                rid2idx = {rid:i for i,rid in enumerate(rna_ids)}
                train_data.pos_edge_label_index = build_edge_index(train_pos, pid2idx, rid2idx)
                test_data.pos_edge_label_index  = build_edge_index(test_pos, pid2idx, rid2idx)


                if neg_strategy == "none":

                    train_data.neg_edge_label_index = filter_bipartite_negatives(
                        train_data.neg_edge_label_index, data_all.num_rna)
                    test_data.neg_edge_label_index  = filter_bipartite_negatives(
                        test_data.neg_edge_label_index,  data_all.num_rna)
                else:

                    train_neg_file = os.path.join(base_path, dataset_name,
                                                  f"fold{fold}", "train", "negative_pairs.txt")
                    test_neg_file  = os.path.join(base_path, dataset_name,
                                                  f"fold{fold}", "test",  "negative_pairs.txt")
                    num_pos_train = train_data.pos_edge_label_index.size(1)
                    num_pos_test  = test_data.pos_edge_label_index.size(1)

                    train_data.neg_edge_label_index = get_negative_samples(
                        train_neg_file, data_all, data_all.num_rna,
                        num_pos_train, train_data.pos_edge_label_index.device,
                        strategy=neg_strategy
                    )
                    test_data.neg_edge_label_index  = get_negative_samples(
                        test_neg_file,  data_all, data_all.num_rna,
                        num_pos_test,  test_data.pos_edge_label_index.device,
                        strategy=neg_strategy
                    )


                encoder = GNNEncoder(in_channels=output_dim, hidden_channels=64, out_channels=128, heads=8)
                edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
                degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
                model = RPI(encoder, edge_decoder, degree_decoder, Mask(p=0.4))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)


                metrics, best_state = train_and_evaluate(model, train_data, test_data, optimizer)


                result_row = {
                    "Conv_Layer1": conv_combo[0],
                    "Conv_Layer2": conv_combo[1],
                    "Neg_Sampling": neg_strategy,
                    "Dataset": dataset_name,
                    "Fold": fold,
                    "AUC": metrics[0],
                    "AP": metrics[1],
                    "ACC": metrics[2],
                    "SEN": metrics[3],
                    "PRE": metrics[4],
                    "SPE": metrics[5],
                    "F1":  metrics[6],
                    "MCC": metrics[7]
                }
                combo_results.append(result_row)
                all_results.append(result_row)

            combo_df = pd.DataFrame(combo_results)
            combo_csv = os.path.join(save_path_conv,
                                     f"results_{conv_combo[0]}_{conv_combo[1]}_{neg_strategy}.csv")
            combo_df.to_csv(combo_csv, index=False)
            print(f"结果已保存: {combo_csv}")


    all_df = pd.DataFrame(all_results)
    all_csv = os.path.join(save_path_conv, "non_neg_sampling3.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"\n✅ 所有组合结果汇总已保存到: {all_csv}")
