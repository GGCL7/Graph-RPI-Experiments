

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, \
    f1_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
    pro_seqs = {pid: seq for pid, seq in zip(pro_ids, pro_seqs_list)}

    rna_ids = parse_ids(rna_file)
    rna_seqs_list = read_fasta(rna_file)
    rna_seqs = {rid: seq for rid, seq in zip(rna_ids, rna_seqs_list)}
    return pro_seqs, rna_seqs


def load_positive_pairs(file_path):

    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            protein_id, rna_id = parts[0].upper(), parts[1].upper()
            pairs.append((rna_id, protein_id))
    return pairs



class ProEncoder:
    elements = 'AIYHRDC'  # 7种字母

    def __init__(self, WINDOW=3):
        self.WINDOW = WINDOW

        self.k_mers = [a + b + c for a in self.elements for b in self.elements for c in self.elements]
        self.k_mer_map = {k_mer: idx for idx, k_mer in enumerate(self.k_mers)}

    def encode_protein(self, seq):

        seq = ''.join([ch for ch in seq if ch in self.elements])
        if len(seq) < self.WINDOW:
            return 'Error'
        vec = np.zeros(len(self.k_mers), dtype=np.float32)
        for i in range(len(seq) - self.WINDOW + 1):
            k_mer = seq[i:i + self.WINDOW]
            if k_mer in self.k_mer_map:
                vec[self.k_mer_map[k_mer]] += 1
        if vec.max() > 0:
            vec = vec / vec.max()
        return vec


class RNAEncoder:
    elements = 'AUCG'

    def __init__(self, WINDOW=4):
        self.WINDOW = WINDOW

        self.k_mers = [a + b + c + d for a in self.elements for b in self.elements for c in self.elements for d in
                       self.elements]
        self.k_mer_map = {k_mer: idx for idx, k_mer in enumerate(self.k_mers)}

    def encode_RNA(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([ch for ch in seq if ch in self.elements])
        if len(seq) < self.WINDOW:
            return 'Error'
        vec = np.zeros(len(self.k_mers), dtype=np.float32)
        for i in range(len(seq) - self.WINDOW + 1):
            k_mer = seq[i:i + self.WINDOW]
            if k_mer in self.k_mer_map:
                vec[self.k_mer_map[k_mer]] += 1
        if vec.max() > 0:
            vec = vec / vec.max()
        return vec



def segmented_sequence(seq):
    seq_len = len(seq)
    idx = seq_len // 5
    return seq, seq[:idx], seq[idx:2 * idx], seq[2 * idx:3 * idx], seq[3 * idx:4 * idx], seq[4 * idx:]


def coding_pairs(pairs, pro_seqs, rna_seqs, PE, RE, kind):
    samples = []
    for rp in tqdm(pairs):
        if rp[0] in rna_seqs and rp[1] in pro_seqs:
            r_seq = rna_seqs[rp[0]]
            p_seq = pro_seqs[rp[1]]

            r_seq_full, r1, r2, r3, r4, r5 = segmented_sequence(r_seq)
            p_seq_full, p1, p2, p3, p4, p5 = segmented_sequence(p_seq)

            p_vec = PE.encode_protein(p_seq_full)
            p_vec1 = PE.encode_protein(p1)
            p_vec2 = PE.encode_protein(p2)
            p_vec3 = PE.encode_protein(p3)
            p_vec4 = PE.encode_protein(p4)
            p_vec5 = PE.encode_protein(p5)
            r_vec = RE.encode_RNA(r_seq_full)
            r_vec1 = RE.encode_RNA(r1)
            r_vec2 = RE.encode_RNA(r2)
            r_vec3 = RE.encode_RNA(r3)
            r_vec4 = RE.encode_RNA(r4)
            r_vec5 = RE.encode_RNA(r5)

            if (isinstance(p_vec, str) or isinstance(p_vec1, str) or isinstance(p_vec2, str) or
                    isinstance(p_vec3, str) or isinstance(p_vec4, str) or isinstance(p_vec5, str) or
                    isinstance(r_vec, str) or isinstance(r_vec1, str) or isinstance(r_vec2, str) or
                    isinstance(r_vec3, str) or isinstance(r_vec4, str) or isinstance(r_vec5, str)):
                continue
            samples.append([[r_vec, r_vec1, r_vec2, r_vec3, r_vec4, r_vec5],
                            [p_vec, p_vec1, p_vec2, p_vec3, p_vec4, p_vec5],
                            kind])
    return samples


def standardize(arr):
    scaler = StandardScaler()
    return scaler.fit_transform(arr)


def pre_process_data(samples, VECTOR_REPETITION_CNN=1, WINDOW_P_UPLIMIT=3, WINDOW_R_UPLIMIT=4):

    r_kmer = np.array([s[0][0] for s in samples])
    r_kmer_1 = np.array([s[0][1] for s in samples])
    r_kmer_2 = np.array([s[0][2] for s in samples])
    r_kmer_3 = np.array([s[0][3] for s in samples])
    r_kmer_4 = np.array([s[0][4] for s in samples])
    r_kmer_5 = np.array([s[0][5] for s in samples])
    p_kmer = np.array([s[1][0] for s in samples])
    p_kmer_1 = np.array([s[1][1] for s in samples])
    p_kmer_2 = np.array([s[1][2] for s in samples])
    p_kmer_3 = np.array([s[1][3] for s in samples])
    p_kmer_4 = np.array([s[1][4] for s in samples])
    p_kmer_5 = np.array([s[1][5] for s in samples])
    y = np.array([s[2] for s in samples])


    r_kmer = standardize(r_kmer)
    r_kmer_1 = standardize(r_kmer_1)
    r_kmer_2 = standardize(r_kmer_2)
    r_kmer_3 = standardize(r_kmer_3)
    r_kmer_4 = standardize(r_kmer_4)
    r_kmer_5 = standardize(r_kmer_5)
    p_kmer = standardize(p_kmer)
    p_kmer_1 = standardize(p_kmer_1)
    p_kmer_2 = standardize(p_kmer_2)
    p_kmer_3 = standardize(p_kmer_3)
    p_kmer_4 = standardize(p_kmer_4)
    p_kmer_5 = standardize(p_kmer_5)


    def expand_dim(x):
        return np.expand_dims(x, axis=1)

    r_kmer = expand_dim(r_kmer)
    r_kmer_1 = expand_dim(r_kmer_1)
    r_kmer_2 = expand_dim(r_kmer_2)
    r_kmer_3 = expand_dim(r_kmer_3)
    r_kmer_4 = expand_dim(r_kmer_4)
    r_kmer_5 = expand_dim(r_kmer_5)
    p_kmer = expand_dim(p_kmer)
    p_kmer_1 = expand_dim(p_kmer_1)
    p_kmer_2 = expand_dim(p_kmer_2)
    p_kmer_3 = expand_dim(p_kmer_3)
    p_kmer_4 = expand_dim(p_kmer_4)
    p_kmer_5 = expand_dim(p_kmer_5)
    return (p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4, p_kmer_5,
            r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4, r_kmer_5, y)



class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def compute_metrics(y_true, y_pred, y_scores):
    try:
        auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) == 2 else 0.0
    except Exception:
        auc = 0.0
    try:
        ap = average_precision_score(y_true, y_scores)
    except Exception:
        ap = 0.0
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)  # 敏感性
    pre = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return auc, ap, acc, sen, pre, spe, f1, mcc



if __name__ == "__main__":
    # 数据集及路径设置
    dataset_names = ["PRImerged"]

    data_base = "Data"
    output_dir = "LPI-SKMSC"  # 输出文件夹
    os.makedirs(output_dir, exist_ok=True)


    all_fold_metrics = []


    for ds in dataset_names:
        dataset_folder = os.path.join(data_base, ds)
        protein_file = os.path.join(dataset_folder, "Protein.txt")
        rna_file = os.path.join(dataset_folder, "RNA.txt")


        ds_output_folder = os.path.join(output_dir, ds)
        os.makedirs(ds_output_folder, exist_ok=True)


        pro_seqs, rna_seqs = load_sequences(protein_file, rna_file)

        PE = ProEncoder(WINDOW=3)
        RE = RNAEncoder(WINDOW=4)


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



            train_samples = coding_pairs(train_pos_pairs, pro_seqs, rna_seqs, PE, RE, kind=1)
            train_samples += coding_pairs(train_neg_pairs, pro_seqs, rna_seqs, PE, RE, kind=0)
            test_samples = coding_pairs(test_pos_pairs, pro_seqs, rna_seqs, PE, RE, kind=1)
            test_samples += coding_pairs(test_neg_pairs, pro_seqs, rna_seqs, PE, RE, kind=0)

            (p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4, p_kmer_5,
             r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4, r_kmer_5, y_train) = pre_process_data(
                train_samples, VECTOR_REPETITION_CNN=1, WINDOW_P_UPLIMIT=3, WINDOW_R_UPLIMIT=4)



            def flatten(x):
                return x.reshape(x.shape[0], -1)


            X_list = [flatten(arr) for arr in [p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4, p_kmer_5,
                                               r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4, r_kmer_5]]
            X_train = np.concatenate(X_list, axis=1)

            (p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4, p_kmer_5,
             r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4, r_kmer_5, y_test) = pre_process_data(
                test_samples, VECTOR_REPETITION_CNN=1, WINDOW_P_UPLIMIT=3, WINDOW_R_UPLIMIT=4)
            X_list = [flatten(arr) for arr in [p_kmer, p_kmer_1, p_kmer_2, p_kmer_3, p_kmer_4, p_kmer_5,
                                               r_kmer, r_kmer_1, r_kmer_2, r_kmer_3, r_kmer_4, r_kmer_5]]
            X_test = np.concatenate(X_list, axis=1)



            train_dataset = SimpleDataset(X_train, y_train)
            test_dataset = SimpleDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


            input_dim = X_train.shape[1]
            model = SimpleMLP(input_dim, hidden_dim=256, num_classes=2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            num_epochs = 100
            best_auc = 0.0
            best_metrics = None
            best_epoch = 0


            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * batch_X.size(0)
                epoch_loss = running_loss / len(train_dataset)


                model.eval()
                all_preds = []
                all_labels = []
                all_probs = []
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(device)
                        outputs = model(batch_X)

                        probs = torch.softmax(outputs, dim=1)

                        pos_probs = probs[:, 1].cpu().numpy()
                        _, predicted = torch.max(outputs, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(batch_y.numpy())
                        all_probs.extend(pos_probs)

                auc, ap, acc, sen, pre, spe, f1, mcc = compute_metrics(all_labels, all_preds, all_probs)
                print("Fold {} Epoch [{}/{}], Loss: {:.4f}, AUC: {:.3f}, ACC: {:.3f}".format(
                    fold, epoch + 1, num_epochs, epoch_loss, auc, acc))


                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + 1
                    best_metrics = {
                        "Fold": fold,
                        "AUC": float(f"{auc:.3f}"),
                        "AP": float(f"{ap:.3f}"),
                        "ACC": float(f"{acc:.3f}"),
                        "SEN": float(f"{sen:.3f}"),
                        "PRE": float(f"{pre:.3f}"),
                        "SPE": float(f"{spe:.3f}"),
                        "F1": float(f"{f1:.3f}"),
                        "MCC": float(f"{mcc:.3f}")
                    }

                    model_path = os.path.join(ds_output_folder, f"best_model_fold{fold}.pt")
                    torch.save(model.state_dict(), model_path)


            all_fold_metrics.append(best_metrics)


        metrics_names = ["AUC", "AP", "ACC", "SEN", "PRE", "SPE", "F1", "MCC"]
        summary = {}
        for m in metrics_names:
            values = [fold_metric[m] for fold_metric in all_fold_metrics]
            summary[m] = {
                "mean": float(f"{np.mean(values):.3f}"),
                "std": float(f"{np.std(values):.3f}")
            }

        results_path = os.path.join(ds_output_folder, "cross_val_results.txt")
        with open(results_path, "w") as f:
            f.write("每折指标：\n")
            for metric in all_fold_metrics:
                f.write(str(metric) + "\n")
            f.write("\n五折指标平均值及标准差：\n")
            for m in metrics_names:
                f.write(f"{m}: mean = {summary[m]['mean']}, std = {summary[m]['std']}\n")
        print("\n五折交叉验证结果：")
        for m in metrics_names:
            print(f"{m}: mean = {summary[m]['mean']}, std = {summary[m]['std']}")

