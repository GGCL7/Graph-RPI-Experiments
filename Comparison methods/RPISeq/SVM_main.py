import os
import itertools
import numpy as np
import pandas as pd
import joblib
from math import ceil, floor
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix)


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


reduced_map = {
    'A': '0', 'G': '0', 'V': '0',
    'I': '1', 'L': '1', 'F': '1', 'P': '1',
    'Y': '2', 'M': '2', 'T': '2', 'S': '2',
    'H': '3', 'N': '3', 'Q': '3', 'W': '3',
    'R': '4', 'K': '4',
    'D': '5', 'E': '5',
    'C': '6'
}

def reduce_protein(seq):
    seq = seq.upper()
    reduced = ""
    for aa in seq:
        if aa in reduced_map:
            reduced += reduced_map[aa]
    return reduced

def get_protein_feature(seq):
    reduced_seq = reduce_protein(seq)
    n = len(reduced_seq)
    total = n - 2 if n >= 3 else 0
    symbols = ['0', '1', '2', '3', '4', '5', '6']
    all_triads = [''.join(p) for p in itertools.product(symbols, repeat=3)]
    triads = {t: 0 for t in all_triads}
    if total > 0:
        for i in range(n - 2):
            triad = reduced_seq[i:i + 3]
            if len(triad) == 3:
                triads[triad] += 1
    feature = [triads[t] / total if total > 0 else 0 for t in all_triads]
    return np.array(feature)


def get_rna_feature(seq):
    seq = seq.upper().replace('T', 'U')
    n = len(seq)
    total = n - 3 if n >= 4 else 0
    nucleotides = ['A', 'C', 'G', 'U']
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=4)]
    kmers_count = {k: 0 for k in all_kmers}
    if total > 0:
        for i in range(n - 3):
            kmer = seq[i:i + 4]
            if kmer in kmers_count:
                kmers_count[kmer] += 1
    feature = [kmers_count[k] / total if total > 0 else 0 for k in all_kmers]
    return np.array(feature)


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


def extract_features(pairs, rna_seqs, pro_seqs):
    features = []
    for rna_id, pro_id in pairs:
        rna_seq = rna_seqs.get(rna_id, "")
        pro_seq = pro_seqs.get(pro_id, "")
        rna_feat = get_rna_feature(rna_seq)
        pro_feat = get_protein_feature(pro_seq)
        feat = np.concatenate([rna_feat, pro_feat])
        features.append(feat)
    return np.array(features)


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)
    return (round(accuracy, 3), round(sensitivity, 3), round(precision, 3),
            round(specificity, 3), round(f1, 3), round(mcc, 3))


def main():

    dataset_names = ["PRImerged"]
    data_base = "RPISeqSVM"
    output_dir = "RPISeqSVM"
    os.makedirs(output_dir, exist_ok=True)

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



            train_pairs = train_pos_pairs + train_neg_pairs
            test_pairs = test_pos_pairs + test_neg_pairs
            y_train = np.array([1] * len(train_pos_pairs) + [0] * len(train_neg_pairs))
            y_test  = np.array([1] * len(test_pos_pairs)  + [0] * len(test_neg_pairs))


            X_train = extract_features(train_pairs, rna_seqs, pro_seqs)
            X_test  = extract_features(test_pairs, rna_seqs, pro_seqs)


            svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            svm_model.fit(X_train, y_train)
            y_pred = svm_model.predict(X_test)
            try:
                y_scores = svm_model.predict_proba(X_test)[:, 1]
            except Exception:
                y_scores = y_pred

            auc = roc_auc_score(y_test, y_scores) if len(set(y_test)) == 2 else 0.0
            ap = average_precision_score(y_test, y_scores)
            acc = accuracy_score(y_test, y_pred)
            sen = recall_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
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

            model_path = os.path.join(ds_output_folder, f"svm_model_{ds}_fold{fold}.pkl")
            joblib.dump(svm_model, model_path)

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
