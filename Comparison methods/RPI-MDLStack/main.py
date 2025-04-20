


import os
import re
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score, matthews_corrcoef,
                             confusion_matrix)



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)



def parse_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                ids.append(line[1:].strip().upper())
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


def parse_pairs(file_path):
    positive_pairs = []
    negative_pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            protein_id, rna_id, label = line.strip().split('\t')
            protein_id = protein_id.strip().upper()
            rna_id = rna_id.strip().upper()
            if int(label) == 1:
                positive_pairs.append((protein_id, rna_id))
            elif int(label) == 0:
                negative_pairs.append((protein_id, rna_id))
    return positive_pairs, negative_pairs


def extract_features_from_pairs(protein_features, rna_features, positive_pairs, negative_pairs):

    X = []
    y = []

    for protein_id, rna_id in positive_pairs:
        protein_row = protein_features[protein_features['ID'] == protein_id]
        rna_row = rna_features[rna_features['ID'] == rna_id]
        if not protein_row.empty and not rna_row.empty:
            protein_feat = protein_row.drop(columns=['ID']).values.flatten()
            rna_feat = rna_row.drop(columns=['ID']).values.flatten()
            features = np.concatenate((protein_feat, rna_feat), axis=0)
            X.append(features)
            y.append(1)

    for protein_id, rna_id in negative_pairs:
        protein_row = protein_features[protein_features['ID'] == protein_id]
        rna_row = rna_features[rna_features['ID'] == rna_id]
        if not protein_row.empty and not rna_row.empty:
            protein_feat = protein_row.drop(columns=['ID']).values.flatten()
            rna_feat = rna_row.drop(columns=['ID']).values.flatten()
            features = np.concatenate((protein_feat, rna_feat), axis=0)
            X.append(features)
            y.append(0)
    return np.array(X), np.array(y)



def feature_selection_lasso(X, y, alpha=0.01, target_dim=200):
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)
    coef = np.abs(lasso.coef_)
    selected_idx = np.argsort(coef)[-target_dim:]
    X_selected = X[:, selected_idx]
    return X_selected, selected_idx


def train_mlp(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print("MLP training completed.")
    return mlp, accuracy_score(y_test, y_pred)


def train_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("RF training completed.")
    return rf, accuracy_score(y_test, y_pred)


def train_et(X_train, y_train, X_test, y_test):
    et = ExtraTreesClassifier(n_estimators=100)
    et.fit(X_train, y_train)
    y_pred = et.predict(X_test)
    print("ET training completed.")
    return et, accuracy_score(y_test, y_pred)


class GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def train_gru(X_train, y_train, X_test, y_test):
    model = GRU_Model(input_size=X_train.shape[1], hidden_size=64, output_size=1)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_3d = torch.tensor(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_3d = torch.tensor(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), dtype=torch.float32, device=device)

    print("Starting GRU training...")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_3d)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"GRU Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_3d)
        y_pred = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy()
    print("GRU training completed.")
    return model, accuracy_score(y_test, y_pred)


class DNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def train_dnn(X_train, y_train, X_test, y_test):
    model = DNN_Model(input_size=X_train.shape[1], hidden_size=128, output_size=1)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    print("Starting DNN training...")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"DNN Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32, device=device))
        y_pred = (outputs.squeeze() > 0.5).cpu().numpy()
    print("DNN training completed.")
    return model, accuracy_score(y_test, y_pred)


def stacking_model(X_train, y_train, X_test, y_test, models):
    base_preds_train = []
    base_preds_test = []
    for model in models:
        if isinstance(model, nn.Module):
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                train_pred = model(X_train_tensor).squeeze().cpu().numpy()
                test_pred = model(X_test_tensor).squeeze().cpu().numpy()
        else:
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
        base_preds_train.append(train_pred)
        base_preds_test.append(test_pred)
    base_preds_train = np.column_stack(base_preds_train)
    base_preds_test = np.column_stack(base_preds_test)

    et_stack = ExtraTreesClassifier(n_estimators=100)
    et_stack.fit(base_preds_train, y_train)
    y_pred_stack = et_stack.predict(base_preds_test)
    return et_stack, accuracy_score(y_test, y_pred_stack)


def get_stacking_features(X, base_models):
    base_preds = []
    for model in base_models:
        if isinstance(model, nn.Module):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                pred = model(X_tensor).squeeze().cpu().numpy()
        else:
            pred = model.predict(X)
        base_preds.append(pred)
    return np.column_stack(base_preds)


def train_and_save_best_models(X_train, y_train, X_test, y_test):
    models = {}
    mlp_model, mlp_acc = train_mlp(X_train, y_train, X_test, y_test)
    rf_model, rf_acc = train_rf(X_train, y_train, X_test, y_test)
    et_model, et_acc = train_et(X_train, y_train, X_test, y_test)
    gru_model, gru_acc = train_gru(X_train, y_train, X_test, y_test)
    dnn_model, dnn_acc = train_dnn(X_train, y_train, X_test, y_test)
    models['MLP'] = mlp_model
    models['RF'] = rf_model
    models['ET'] = et_model
    models['GRU'] = gru_model
    models['DNN'] = dnn_model

    print("Saving individual models...")
    for model_name, model in models.items():
        joblib.dump(model, f"{model_name}_model.pkl")

    base_models = [mlp_model, rf_model, et_model, gru_model, dnn_model]
    stack_model, stack_acc = stacking_model(X_train, y_train, X_test, y_test, base_models)
    joblib.dump(stack_model, 'old/stacked_model.pkl')
    print(f"Stacking model saved with accuracy: {stack_acc:.4f}")
    return stack_model, base_models



if __name__ == "__main__":
    dataset_names = ["RPImerged"]
    data_base = "RPI-MDLStack"
    output_dir = "RPI-MDLStack"

    for ds in dataset_names:
        ds_original_folder = os.path.join(data_base, ds)
        ds_feature_folder = os.path.join(output_dir, ds)
        protein_features = pd.read_csv(os.path.join(ds_feature_folder, "protein_features.csv"))
        rna_features = pd.read_csv(os.path.join(ds_feature_folder, "rna_features.csv"))
        print("Protein features shape:", protein_features.shape)
        print("RNA features shape:", rna_features.shape)

        all_fold_metrics = []


        for fold in range(1, 6):
            fold_dir = os.path.join(ds_original_folder, f"fold{fold}")
            train_pos_file = os.path.join(fold_dir, "train", "train_pairs.txt")
            test_pos_file = os.path.join(fold_dir, "test", "test_pairs.txt")
            train_neg_file = os.path.join(os.path.dirname(train_pos_file), "new_negative_pairs.txt")
            test_neg_file = os.path.join(os.path.dirname(test_pos_file), "new_negative_pairs.txt")


            train_pos_pairs = load_positive_pairs(train_pos_file)
            test_pos_pairs = load_positive_pairs(test_pos_file)
            train_neg_pairs = load_positive_pairs(train_neg_file)
            test_neg_pairs = load_positive_pairs(test_neg_file)

            X_train, y_train = extract_features_from_pairs(protein_features, rna_features, train_pos_pairs, train_neg_pairs)
            X_test, y_test = extract_features_from_pairs(protein_features, rna_features, test_pos_pairs, test_neg_pairs)

            best_model, base_models = train_and_save_best_models(X_train, y_train, X_test, y_test)

            stack_X_test = get_stacking_features(X_test, base_models)
            y_pred = best_model.predict(stack_X_test)
            try:
                y_scores = best_model.decision_function(stack_X_test)
            except Exception:
                y_scores = best_model.predict_proba(stack_X_test)[:, 1]
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
            all_fold_metrics.append(metrics)


        metrics_names = ["AUC", "AP", "ACC", "SEN", "PRE", "SPE", "F1", "MCC"]
        summary = {}
        for m in metrics_names:
            values = [fold_metric[m] for fold_metric in all_fold_metrics]
            summary[m] = {"mean": np.mean(values), "std": np.std(values)}
        results_path = os.path.join(ds_feature_folder, "cross_val_results.txt")
        with open(results_path, "w") as f:
            f.write("每折指标：\n")
            for metric in all_fold_metrics:
                f.write(str(metric) + "\n")
            f.write("\n五折指标平均值及标准差：\n")
            for m in metrics_names:
                f.write(f"{m}: mean = {summary[m]['mean']:.3f}, std = {summary[m]['std']:.3f}\n")
        print(f"\n数据集 {ds} 五折交叉验证结果：")
        for m in metrics_names:
            print(f"{m}: mean = {summary[m]['mean']:.3f}, std = {summary[m]['std']:.3f}")


