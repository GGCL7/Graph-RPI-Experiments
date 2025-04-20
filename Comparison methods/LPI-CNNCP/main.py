


import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix)


protein_letters = "ACDEFGHIKLMNPQRSTVWY"
rna_letters = "ACGU"

def build_vocab(letters):
    vocab = {letter: idx + 1 for idx, letter in enumerate(letters)}
    vocab['UNK'] = len(vocab) + 1
    return vocab

protein_vocab = build_vocab(protein_letters)
rna_vocab = build_vocab(rna_letters)

def encode_sequence(seq, vocab, max_len):
    indices = [vocab.get(char, vocab['UNK']) for char in seq]
    if len(indices) < max_len:
        indices = indices + [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices
def load_positive_pairs(file_path):

    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            protein_id, rna_id = parts[0].upper(), parts[1].upper()
            pairs.append((protein_id, rna_id))
    return pairs


def parse_ids(file_path):
    ids = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                ids.append(line[1:].strip())
    return ids

def read_fasta(file_path):
    sequences = []
    sequence = ''
    with open(file_path, 'r') as f:
        for line in f:
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

def load_sequences(protein_file, rna_file):
    pro_ids = parse_ids(protein_file)
    pro_seqs_list = read_fasta(protein_file)
    pro_seqs = {pid.upper(): seq for pid, seq in zip(pro_ids, pro_seqs_list)}
    rna_ids = parse_ids(rna_file)
    rna_seqs_list = read_fasta(rna_file)
    rna_seqs = {rid.upper(): seq for rid, seq in zip(rna_ids, rna_seqs_list)}
    return pro_seqs, rna_seqs


def parse_pairs(file_path):

    positive_pairs = []
    negative_pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            protein_id, rna_id, label = line.strip().split('\t')
            if int(label) == 1:
                positive_pairs.append((protein_id.upper(), rna_id.upper()))
            elif int(label) == 0:
                negative_pairs.append((protein_id.upper(), rna_id.upper()))
    return positive_pairs, negative_pairs


def load_negative_pairs(file_path):
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            protein_id, rna_id = parts[0].upper(), parts[1].upper()
            pairs.append((protein_id, rna_id))
    return pairs


class RPIDatasetFold(Dataset):
    def __init__(self, protein_dict, rna_dict, pos_pairs, neg_pairs, protein_max_len=1000, rna_max_len=3500):
        """
        pos_pairs、neg_pairs：格式均为 (protein_id, rna_id)
        """
        self.protein_dict = protein_dict
        self.rna_dict = rna_dict
        self.pairs = []
        for pair in pos_pairs:
            self.pairs.append((pair[0], pair[1], 1))
        for pair in neg_pairs:
            self.pairs.append((pair[0], pair[1], 0))
        self.protein_max_len = protein_max_len
        self.rna_max_len = rna_max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        protein_id, rna_id, label = self.pairs[idx]
        protein_seq = self.protein_dict.get(protein_id, "")
        rna_seq = self.rna_dict.get(rna_id, "")
        protein_encoded = encode_sequence(protein_seq, protein_vocab, self.protein_max_len)
        rna_encoded = encode_sequence(rna_seq, rna_vocab, self.rna_max_len)
        return (torch.tensor(protein_encoded, dtype=torch.long),
                torch.tensor(rna_encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))


class RPICNN(nn.Module):
    def __init__(self,
                 protein_vocab_size, rna_vocab_size,
                 protein_embed_dim=15, rna_embed_dim=21,
                 protein_max_len=1000, rna_max_len=3500,
                 num_filters=96, fc_dim=96, dropout_rate=0.3):
        super(RPICNN, self).__init__()
        self.protein_embedding = nn.Embedding(protein_vocab_size, protein_embed_dim, padding_idx=0)
        self.rna_embedding = nn.Embedding(rna_vocab_size, rna_embed_dim, padding_idx=0)
        self.protein_conv = nn.Conv2d(in_channels=1,
                                      out_channels=num_filters,
                                      kernel_size=(49, protein_embed_dim))
        self.protein_bn = nn.BatchNorm2d(num_filters)
        self.rna_conv = nn.Conv2d(in_channels=1,
                                  out_channels=num_filters,
                                  kernel_size=(64, rna_embed_dim),
                                  stride=(1, 1))
        self.rna_bn = nn.BatchNorm2d(num_filters)
        self.fc = nn.Linear(num_filters * 2, fc_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(fc_dim, 2)

    def forward(self, protein_input, rna_input):
        p_embed = self.protein_embedding(protein_input)
        p_embed = p_embed.unsqueeze(1)
        p_conv = self.protein_conv(p_embed)
        p_conv = self.protein_bn(p_conv)
        p_conv = torch.relu(p_conv)
        p_pool = torch.max(p_conv, dim=2)[0].squeeze(2)

        r_embed = self.rna_embedding(rna_input)
        r_embed = r_embed.unsqueeze(1)
        r_conv = self.rna_conv(r_embed)
        r_conv = self.rna_bn(r_conv)
        r_conv = torch.relu(r_conv)
        r_pool = torch.max(r_conv, dim=2)[0].squeeze(2)

        x = torch.cat([p_pool, r_pool], dim=1)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.out(x)
        return logits

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for protein, rna, label in dataloader:
        protein = protein.to(device)
        rna = rna.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(protein, rna)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * protein.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for protein, rna, label in dataloader:
            protein = protein.to(device)
            rna = rna.to(device)
            label = label.to(device)
            outputs = model(protein, rna)
            loss = criterion(outputs, label)
            total_loss += loss.item() * protein.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == label).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

def main():

    dataset_names = ["RPI2241"]
    data_base = "Data"
    output_dir = "LPI-CNNCP"
    os.makedirs(output_dir, exist_ok=True)


    PROTEIN_MAX_LEN = 1000
    RNA_MAX_LEN = 3500
    num_epochs = 100
    batch_size = 16
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    all_fold_metrics = []
    for ds in dataset_names:

        dataset_folder = os.path.join(data_base, ds)
        protein_file = os.path.join(dataset_folder, "Protein.txt")
        rna_file = os.path.join(dataset_folder, "RNA.txt")


        ds_output_folder = os.path.join(output_dir, ds)
        os.makedirs(ds_output_folder, exist_ok=True)


        pro_seqs, rna_seqs = load_sequences(protein_file, rna_file)

        fold_results = []
        for fold in range(1, 6):

            fold_dir = os.path.join(dataset_folder, f"fold{fold}")

            train_pos_file = os.path.join(fold_dir, "train", "train_pairs.txt")
            test_pos_file = os.path.join(fold_dir, "test", "test_pairs.txt")

            train_neg_file = os.path.join(os.path.dirname(train_pos_file), "new_negative_pairs.txt")
            test_neg_file = os.path.join(os.path.dirname(test_pos_file), "new_negative_pairs.txt")

            train_pos_pairs = load_positive_pairs(train_pos_file)
            test_pos_pairs = load_positive_pairs(test_pos_file)
            train_neg_pairs = load_positive_pairs(train_neg_file)
            test_neg_pairs = load_positive_pairs(test_neg_file)


            train_dataset = RPIDatasetFold(pro_seqs, rna_seqs, train_pos_pairs, train_neg_pairs,
                                           PROTEIN_MAX_LEN, RNA_MAX_LEN)
            test_dataset = RPIDatasetFold(pro_seqs, rna_seqs, test_pos_pairs, test_neg_pairs,
                                          PROTEIN_MAX_LEN, RNA_MAX_LEN)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


            model = RPICNN(protein_vocab_size=len(protein_vocab) + 1,
                           rna_vocab_size=len(rna_vocab) + 1,
                           protein_embed_dim=15,
                           rna_embed_dim=21,
                           protein_max_len=PROTEIN_MAX_LEN,
                           rna_max_len=RNA_MAX_LEN,
                           num_filters=96,
                           fc_dim=96,
                           dropout_rate=0.3).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_fold_acc = 0.0
            best_state = None
            for epoch in range(num_epochs):
                train_loss = train_model(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
                print(f"Fold {fold} Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f}")
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    best_state = model.state_dict()

            model.load_state_dict(best_state)
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)


            model_path = os.path.join(ds_output_folder, f"best_model_{ds}_fold{fold}.pth")
            torch.save(model.state_dict(), model_path)
            all_preds = []
            all_labels = []
            all_probs = []
            model.eval()
            with torch.no_grad():
                for protein, rna, label in test_loader:
                    protein = protein.to(device)
                    rna = rna.to(device)
                    outputs = model(protein, rna)
                    probs = torch.softmax(outputs, dim=1)[:, 1]  # 正类概率
                    preds = torch.argmax(outputs, dim=1)
                    all_labels.extend(label.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())


            auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.0
            ap = average_precision_score(all_labels, all_probs)
            acc = accuracy_score(all_labels, all_preds)
            sen = recall_score(all_labels, all_preds)
            pre = precision_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            mcc = matthews_corrcoef(all_labels, all_preds)
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            metrics = {
                "Fold": fold,
                "AUC": auc,
                "AP": ap,
                "ACC": acc,
                "SEN": sen,
                "PRE": pre,
                "SPE": spe,
                "F1": f1,
                "MCC": mcc
            }
            print(f"Fold {fold} metrics: {metrics}")
            fold_results.append(metrics)
            all_fold_metrics.append(metrics)


        results_df = pd.DataFrame(fold_results)
        results_csv = os.path.join(ds_output_folder, f"fold_results_{ds}.csv")
        results_df.to_csv(results_csv, index=False)

        metrics_names = ["AUC", "AP", "ACC", "SEN", "PRE", "SPE", "F1", "MCC"]
        summary = {}
        for m in metrics_names:
            values = [fold_metric[m] for fold_metric in all_fold_metrics]
            summary[m] = {"mean": np.mean(values), "std": np.std(values)}
        results_path = os.path.join(ds_output_folder, "cross_val_results.txt")
        with open(results_path, "w") as f:
            f.write("每折指标：\n")
            for metric in all_fold_metrics:
                f.write(str(metric) + "\n")
            f.write("\n五折指标平均值及标准差：\n")
            for m in metrics_names:
                f.write(f"{m}: mean = {summary[m]['mean']:.3f}, std = {summary[m]['std']:.3f}\n")
        print("\n五折交叉验证结果：")
        for m in metrics_names:
            print(f"{m}: mean = {summary[m]['mean']:.3f}, std = {summary[m]['std']:.3f}")

if __name__ == "__main__":
    main()


