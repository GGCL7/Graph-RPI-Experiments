
# import os
# import re
# import math
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# import torch.optim as optim
# from torch import nn
# from sklearn.model_selection import KFold
# from sklearn.svm import SVC
# from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Lasso
# from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
#                              recall_score, precision_score, f1_score, matthews_corrcoef,
#                              confusion_matrix)
# from sklearn.manifold import MDS
# import iFeatureOmegaCLI
#
# # 优先使用 mps 加速（macOS 用户），否则使用 CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print("Using device:", device)
#
#
# ########################################
# # 辅助函数：读取ID、FASTA序列及交互对文件
# ########################################
# def parse_ids(file_path):
#     ids = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             if line.startswith('>'):
#                 ids.append(line[1:].strip().upper())
#     return ids
#
#
# def read_fasta(file_path):
#     sequences = []
#     sequence = ''
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             if line.startswith('>'):
#                 if sequence:
#                     sequences.append(sequence)
#                     sequence = ''
#             else:
#                 sequence += line
#         if sequence:
#             sequences.append(sequence)
#     return sequences
#
#
# def load_positive_pairs(file_path):
#     """
#     读取正样本，每行至少包含两列，
#     返回 (protein_id, rna_id) 列表
#     """
#     pairs = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) < 2:
#                 continue
#             protein_id, rna_id = parts[0].upper(), parts[1].upper()
#             pairs.append((protein_id, rna_id))
#     return pairs
#
#
# def parse_pairs(file_path):
#     """
#     读取交互对文件，每行格式为：protein_id <tab> rna_id <tab> label，
#     根据 label 将正负样本分别返回
#     """
#     positive_pairs = []
#     negative_pairs = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             protein_id, rna_id, label = line.strip().split('\t')
#             protein_id = protein_id.strip().upper()
#             rna_id = rna_id.strip().upper()
#             if int(label) == 1:
#                 positive_pairs.append((protein_id, rna_id))
#             elif int(label) == 0:
#                 negative_pairs.append((protein_id, rna_id))
#     return positive_pairs, negative_pairs
#
#
# def extract_features_from_pairs(protein_features, rna_features, positive_pairs, negative_pairs):
#     """
#     根据交互对，从蛋白质和 RNA 特征（CSV 文件中包含 ID 列）中提取对应样本的特征，
#     若匹配到多个记录则取第一行
#     """
#     X = []
#     y = []
#     # 正样本
#     for protein_id, rna_id in positive_pairs:
#         protein_row = protein_features[protein_features['ID'] == protein_id]
#         rna_row = rna_features[rna_features['ID'] == rna_id]
#         if not protein_row.empty and not rna_row.empty:
#             protein_feat = protein_row.drop(columns=['ID']).iloc[0].values.flatten()
#             rna_feat = rna_row.drop(columns=['ID']).iloc[0].values.flatten()
#             features = np.concatenate((protein_feat, rna_feat), axis=0)
#             X.append(features)
#             y.append(1)
#     # 负样本
#     for protein_id, rna_id in negative_pairs:
#         protein_row = protein_features[protein_features['ID'] == protein_id]
#         rna_row = rna_features[rna_features['ID'] == rna_id]
#         if not protein_row.empty and not rna_row.empty:
#             protein_feat = protein_row.drop(columns=['ID']).iloc[0].values.flatten()
#             rna_feat = rna_row.drop(columns=['ID']).iloc[0].values.flatten()
#             features = np.concatenate((protein_feat, rna_feat), axis=0)
#             X.append(features)
#             y.append(0)
#     return np.array(X), np.array(y)
#
#
# ########################################
# # 传统机器学习模型训练（使用 scikit-learn）
# ########################################
# def train_et(X_train, y_train, X_test, y_test):
#     model = ExtraTreesClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print("ET training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# def train_knn(X_train, y_train, X_test, y_test):
#     model = KNeighborsClassifier(n_neighbors=5)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print("KNN training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# def train_mlp_sk(X_train, y_train, X_test, y_test):
#     model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print("MLP (sklearn) training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# def train_nb(X_train, y_train, X_test, y_test):
#     model = GaussianNB()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print("Naive Bayes training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# def train_rf(X_train, y_train, X_test, y_test):
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print("RF training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# def train_svm_sk(X_train, y_train, X_test, y_test):
#     model = SVC(kernel='linear', probability=True, random_state=42)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     print("SVM training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# ########################################
# # 深度学习模型（基于 PyTorch），训练轮数均设为 1 epoch
# ########################################
# # GRU 模型
# class GRU_Model(nn.Module):
#     def __init__(self, input_size, hidden_size=64, output_size=1):
#         super(GRU_Model, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.gru(x)
#         out = self.fc(out[:, -1, :])
#         return out
#
#
# def train_gru_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
#     model = GRU_Model(input_size=X_train.shape[1])
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     model.eval()
#     with torch.no_grad():
#         outputs_test = model(X_test_tensor)
#         preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
#     print("GRU training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# # DNN 模型
# class DNN_Model(nn.Module):
#     def __init__(self, input_size, hidden_size=128, output_size=1):
#         super(DNN_Model, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# def train_dnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
#     model = DNN_Model(input_size=X_train.shape[1])
#     model.to(device)
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(torch.sigmoid(outputs), y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     model.eval()
#     with torch.no_grad():
#         outputs_test = model(X_test_tensor)
#         preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
#     print("DNN training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# # CNN 模型（简单一维卷积）
# class CNN_Model(nn.Module):
#     def __init__(self, input_size, num_classes=1):
#         super(CNN_Model, self).__init__()
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Linear(16, num_classes)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
#
# def train_cnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
#     model = CNN_Model(input_size=X_train.shape[1])
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     model.eval()
#     with torch.no_grad():
#         outputs_test = model(X_test_tensor)
#         preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
#     print("CNN training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# # Capsule 模型（利用 Discriminator 和简化版 CBAM）
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=2):
#         super(CBAM, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x
#
#
# class Discriminator(nn.Module):
#     def __init__(self, input_dim, capsule_dim=16, num_capsules=8):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.relu = nn.ReLU()
#         self.fc_caps = nn.Linear(256, num_capsules * capsule_dim)
#         self.num_capsules = num_capsules
#         self.capsule_dim = capsule_dim
#         self.cbam = CBAM(channels=num_capsules, reduction=2)
#         self.fc_out = nn.Linear(num_capsules * capsule_dim, 1)
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         caps = self.fc_caps(x)
#         caps = caps.view(-1, self.num_capsules, self.capsule_dim)
#         cbam_out = self.cbam(caps)
#         cbam_flat = cbam_out.view(-1, self.num_capsules * self.capsule_dim)
#         out = self.fc_out(cbam_flat)
#         return out
#
#
# def train_capsule_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
#     model = Discriminator(input_dim=X_train.shape[1], capsule_dim=16, num_capsules=8)
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     model.eval()
#     with torch.no_grad():
#         outputs_test = model(X_test_tensor)
#         preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
#     print("Capsule training completed.")
#     return model, accuracy_score(y_test, preds)
#
#
# ########################################
# # 决策融合：Stacking 集成
# ########################################
# def get_base_predictions(models, X):
#     preds = []
#     for model in models:
#         if hasattr(model, "predict"):
#             pred = model.predict(X)
#         else:
#             X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
#             if model.__class__.__name__ == "GRU_Model":
#                 X_tensor = X_tensor.unsqueeze(1)
#             model.eval()
#             with torch.no_grad():
#                 output = model(X_tensor)
#                 pred = (torch.sigmoid(output) > 0.5).cpu().numpy().astype(int).flatten()
#         preds.append(pred)
#     return np.column_stack(preds)
#
#
# def stacking_ensemble(X_train, y_train, X_test, y_test, base_models):
#     base_train_preds = get_base_predictions(base_models, X_train)
#     base_test_preds = get_base_predictions(base_models, X_test)
#     meta_clf = SVC(kernel='linear', probability=True, random_state=42)
#     meta_clf.fit(base_train_preds, y_train)
#     final_preds = meta_clf.predict(base_test_preds)
#     acc = accuracy_score(y_test, final_preds)
#     return meta_clf, acc
#
#
# def train_and_save_best_models(X_train, y_train, X_test, y_test):
#     models = {}
#     mlp_model, _ = train_mlp_sk(X_train, y_train, X_test, y_test)
#     rf_model, _ = train_rf(X_train, y_train, X_test, y_test)
#     et_model, _ = train_et(X_train, y_train, X_test, y_test)
#     gru_model, _ = train_gru_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)
#     dnn_model, _ = train_dnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)
#     cnn_model, _ = train_cnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)
#     capsule_model, _ = train_capsule_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)
#
#     models['MLP'] = mlp_model
#     models['RF'] = rf_model
#     models['ET'] = et_model
#     models['GRU'] = gru_model
#     models['DNN'] = dnn_model
#     models['CNN'] = cnn_model
#     models['Capsule'] = capsule_model
#
#     print("Saving individual models...")
#     for model_name, model in models.items():
#         if hasattr(model, "predict"):
#             joblib.dump(model, f"{model_name}_model.pkl")
#         else:
#             torch.save(model.state_dict(), f"{model_name}_model_pt.pth")
#
#     base_models = [mlp_model, rf_model, et_model, gru_model, dnn_model, cnn_model, capsule_model]
#     stack_model, stack_acc = stacking_ensemble(X_train, y_train, X_test, y_test, base_models)
#     joblib.dump(stack_model, 'stacked_model.pkl')
#     print(f"Stacking model saved with accuracy: {stack_acc:.4f}")
#     return stack_model, base_models
#
#
# ########################################
# # 主程序 —— 利用保存好的特征进行五折交叉验证（使用单一 Pairs.txt 文件）
# # 正样本由 Pairs.txt 随机划分，负样本固定使用各折对应的 new_negative_pairs.txt 文件（每折独立）
# ########################################
# if __name__ == "__main__":
#     # 以 RPI1807 数据集为例，请根据实际情况修改路径
#     ds_feature_folder = "/Users/ggcl7/Desktop/RPI审稿意见新结果/对比之前方法/RPI-capsuleGAN/RPI1807"
#     data_folder = "/Users/ggcl7/Desktop/RPI审稿意见新结果/分层评估对比之前方法/RPI1807"
#     # 从单一 Pairs.txt 文件中提取所有正样本
#     pairs_file = os.path.join("/Users/ggcl7/Desktop/RPI审稿意见新结果/分层评估对比之前方法/RPI1807", "Pairs.txt")
#     pos_pairs_all, _ = parse_pairs(pairs_file)  # 这里只取正样本
#
#     # 读取保存好的蛋白质和 RNA 特征（CSV 文件中包含 ID 列）
#     protein_features = pd.read_csv(os.path.join(ds_feature_folder, "protein_features.csv"))
#     rna_features = pd.read_csv(os.path.join(ds_feature_folder, "rna_features.csv"))
#     print("Protein features shape:", protein_features.shape)
#     print("RNA features shape:", rna_features.shape)
#
#     # 利用 KFold 对所有正样本随机划分 5 折
#     from sklearn.model_selection import KFold
#
#     pos_pairs_all = np.array(pos_pairs_all)
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
#     fold_index = 1
#     all_fold_metrics = []
#
#     for train_idx, test_idx in kf.split(pos_pairs_all):
#         # 正样本随机划分
#         train_pos_pairs = pos_pairs_all[train_idx].tolist()
#         test_pos_pairs = pos_pairs_all[test_idx].tolist()
#         print(f"Fold {fold_index} 正样本：训练 {len(train_pos_pairs)}，测试 {len(test_pos_pairs)}")
#         # 负样本固定：从当前折对应的文件中加载
#         fold_dir = os.path.join(data_folder, f"fold{fold_index}")
#         train_neg_file = os.path.join(fold_dir, "train", "new_negative_pairs.txt")
#         test_neg_file = os.path.join(fold_dir, "test", "new_negative_pairs.txt")
#         train_neg_pairs = load_positive_pairs(train_neg_file)
#         test_neg_pairs = load_positive_pairs(test_neg_file)
#         print(f"Fold {fold_index} 负样本：训练 {len(train_neg_pairs)}，测试 {len(test_neg_pairs)}")
#
#         # 构造训练集和测试集
#         X_train, y_train = extract_features_from_pairs(protein_features, rna_features, train_pos_pairs, train_neg_pairs)
#         X_test, y_test = extract_features_from_pairs(protein_features, rna_features, test_pos_pairs, test_neg_pairs)
#
#         # 训练各基础模型并进行 stacking 融合
#         best_model, base_models = train_and_save_best_models(X_train, y_train, X_test, y_test)
#
#         # 生成 stacking 特征并计算指标
#         base_test_preds = get_base_predictions(base_models, X_test)
#         y_pred = best_model.predict(base_test_preds)
#         try:
#             y_scores = best_model.decision_function(base_test_preds)
#         except Exception:
#             y_scores = best_model.predict_proba(base_test_preds)[:, 1]
#         auc = roc_auc_score(y_test, y_scores) if len(set(y_test)) == 2 else 0.0
#         ap = average_precision_score(y_test, y_scores)
#         acc = accuracy_score(y_test, y_pred)
#         sen = recall_score(y_test, y_pred)
#         pre = precision_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         mcc = matthews_corrcoef(y_test, y_pred)
#         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#         spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#
#         metrics = {
#             "Fold": fold_index,
#             "AUC": auc,
#             "AP": ap,
#             "ACC": acc,
#             "SEN": sen,
#             "PRE": pre,
#             "SPE": spe,
#             "F1": f1,
#             "MCC": mcc
#         }
#         print(f"Fold {fold_index} metrics: {metrics}")
#         all_fold_metrics.append(metrics)
#         fold_index += 1
#
#     # 计算每折指标的均值和标准差，并保存结果
#     metrics_names = ["AUC", "AP", "ACC", "SEN", "PRE", "SPE", "F1", "MCC"]
#     summary = {}
#     for m in metrics_names:
#         values = [fold_metric[m] for fold_metric in all_fold_metrics]
#         summary[m] = {"mean": np.mean(values), "std": np.std(values)}
#     results_path = os.path.join(ds_feature_folder, "cross_val_results.txt")
#     with open(results_path, "w") as f:
#         f.write("每折指标：\n")
#         for metric in all_fold_metrics:
#             f.write(str(metric) + "\n")
#         f.write("\n五折指标平均值及标准差：\n")
#         for m in metrics_names:
#             f.write(f"{m}: mean = {summary[m]['mean']:.3f}, std = {summary[m]['std']:.3f}\n")
#     print("\n五折交叉验证结果：")
#     for m in metrics_names:
#         print(f"{m}: mean = {summary[m]['mean']:.3f}, std = {summary[m]['std']:.3f}")


import os
import re
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score, matthews_corrcoef,
                             confusion_matrix)
from sklearn.manifold import MDS
import iFeatureOmegaCLI

# 优先使用 mps 加速（macOS 用户），否则使用 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


########################################
# 辅助函数：读取ID、FASTA序列及交互对文件
########################################
def parse_ids(file_path):
    ids = []
    with open(file_path, 'r') as f:
        for line in f:
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
    """
    读取正样本，每行至少包含两列，
    返回 (protein_id, rna_id) 列表
    """
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
    """
    读取交互对文件，每行格式为：protein_id <tab> rna_id <tab> label，
    根据 label 返回正样本列表和负样本列表
    """
    positive_pairs = []
    negative_pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            protein_id, rna_id, label = line.strip().split('\t')
            protein_id = protein_id.strip().upper()
            rna_id = rna_id.strip().upper()
            if int(label) == 1:
                positive_pairs.append((protein_id, rna_id))
            elif int(label) == 0:
                negative_pairs.append((protein_id, rna_id))
    return positive_pairs, negative_pairs


def extract_features_from_pairs(protein_features, rna_features, positive_pairs, negative_pairs):
    """
    根据交互对，从蛋白质和RNA特征（CSV 文件中包含 ID 列）中提取对应样本的特征，
    若匹配到多个记录则取第一行
    """
    X = []
    y = []
    # 正样本
    for protein_id, rna_id in positive_pairs:
        protein_row = protein_features[protein_features['ID'] == protein_id]
        rna_row = rna_features[rna_features['ID'] == rna_id]
        if not protein_row.empty and not rna_row.empty:
            protein_feat = protein_row.drop(columns=['ID']).iloc[0].values.flatten()
            rna_feat = rna_row.drop(columns=['ID']).iloc[0].values.flatten()
            features = np.concatenate((protein_feat, rna_feat), axis=0)
            X.append(features)
            y.append(1)
    # 负样本
    for protein_id, rna_id in negative_pairs:
        protein_row = protein_features[protein_features['ID'] == protein_id]
        rna_row = rna_features[rna_features['ID'] == rna_id]
        if not protein_row.empty and not rna_row.empty:
            protein_feat = protein_row.drop(columns=['ID']).iloc[0].values.flatten()
            rna_feat = rna_row.drop(columns=['ID']).iloc[0].values.flatten()
            features = np.concatenate((protein_feat, rna_feat), axis=0)
            X.append(features)
            y.append(0)
    return np.array(X), np.array(y)


########################################
# 传统机器学习模型训练（使用 scikit-learn）
########################################
def train_et(X_train, y_train, X_test, y_test):
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("ET training completed.")
    return model, accuracy_score(y_test, preds)


def train_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("KNN training completed.")
    return model, accuracy_score(y_test, preds)


def train_mlp_sk(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("MLP (sklearn) training completed.")
    return model, accuracy_score(y_test, preds)


def train_nb(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Naive Bayes training completed.")
    return model, accuracy_score(y_test, preds)


def train_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("RF training completed.")
    return model, accuracy_score(y_test, preds)


def train_svm_sk(X_train, y_train, X_test, y_test):
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("SVM training completed.")
    return model, accuracy_score(y_test, preds)


########################################
# 深度学习模型（基于 PyTorch），训练轮数均设为 1 epoch
########################################
# GRU 模型
class GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def train_gru_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
    model = GRU_Model(input_size=X_train.shape[1])
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
    print("GRU training completed.")
    return model, accuracy_score(y_test, preds)


# DNN 模型
class DNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(DNN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_dnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
    model = DNN_Model(input_size=X_train.shape[1])
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(torch.sigmoid(outputs), y_train_tensor)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
    print("DNN training completed.")
    return model, accuracy_score(y_test, preds)


# CNN 模型（简单一维卷积）
class CNN_Model(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_cnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
    model = CNN_Model(input_size=X_train.shape[1])
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
    print("CNN training completed.")
    return model, accuracy_score(y_test, preds)


# Capsule 模型（利用 Discriminator 和简化版 CBAM）
class CBAM(nn.Module):
    def __init__(self, channels, reduction=2):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class Discriminator(nn.Module):
    def __init__(self, input_dim, capsule_dim=16, num_capsules=8):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc_caps = nn.Linear(256, num_capsules * capsule_dim)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.cbam = CBAM(channels=num_capsules, reduction=2)
        self.fc_out = nn.Linear(num_capsules * capsule_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        caps = self.fc_caps(x)
        caps = caps.view(-1, self.num_capsules, self.capsule_dim)
        cbam_out = self.cbam(caps)
        cbam_flat = cbam_out.view(-1, self.num_capsules * self.capsule_dim)
        out = self.fc_out(cbam_flat)
        return out


def train_capsule_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001):
    model = Discriminator(input_dim=X_train.shape[1], capsule_dim=16, num_capsules=8)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        preds = (torch.sigmoid(outputs_test) > 0.5).cpu().numpy().astype(int)
    print("Capsule training completed.")
    return model, accuracy_score(y_test, preds)


########################################
# 决策融合：Stacking 集成
########################################
def get_base_predictions(models, X):
    preds = []
    for model in models:
        if hasattr(model, "predict"):
            pred = model.predict(X)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            if model.__class__.__name__ == "GRU_Model":
                X_tensor = X_tensor.unsqueeze(1)
            model.eval()
            with torch.no_grad():
                output = model(X_tensor)
                pred = (torch.sigmoid(output) > 0.5).cpu().numpy().astype(int).flatten()
        preds.append(pred)
    return np.column_stack(preds)


def stacking_ensemble(X_train, y_train, X_test, y_test, base_models):
    base_train_preds = get_base_predictions(base_models, X_train)
    base_test_preds = get_base_predictions(base_models, X_test)
    meta_clf = SVC(kernel='linear', probability=True, random_state=42)
    meta_clf.fit(base_train_preds, y_train)
    final_preds = meta_clf.predict(base_test_preds)
    acc = accuracy_score(y_test, final_preds)
    return meta_clf, acc


def train_and_save_best_models(X_train, y_train, X_test, y_test):
    models = {}
    mlp_model, _ = train_mlp_sk(X_train, y_train, X_test, y_test)
    rf_model, _ = train_rf(X_train, y_train, X_test, y_test)
    et_model, _ = train_et(X_train, y_train, X_test, y_test)
    gru_model, _ = train_gru_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)
    dnn_model, _ = train_dnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)
    cnn_model, _ = train_cnn_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)
    capsule_model, _ = train_capsule_model(X_train, y_train, X_test, y_test, num_epochs=1, lr=0.001)

    models['MLP'] = mlp_model
    models['RF'] = rf_model
    models['ET'] = et_model
    models['GRU'] = gru_model
    models['DNN'] = dnn_model
    models['CNN'] = cnn_model
    models['Capsule'] = capsule_model

    print("Saving individual models...")
    for model_name, model in models.items():
        if hasattr(model, "predict"):
            joblib.dump(model, f"{model_name}_model.pkl")
        else:
            torch.save(model.state_dict(), f"{model_name}_model_pt.pth")

    base_models = [mlp_model, rf_model, et_model, gru_model, dnn_model, cnn_model, capsule_model]
    stack_model, stack_acc = stacking_ensemble(X_train, y_train, X_test, y_test, base_models)
    joblib.dump(stack_model, 'stacked_model.pkl')
    print(f"Stacking model saved with accuracy: {stack_acc:.4f}")
    return stack_model, base_models


########################################
# 主程序 —— 利用保存好的特征进行五折交叉验证（使用固定的正负样本文件）
# 正样本由各折固定文件 train_pairs.txt 和 test_pairs.txt 提取，
# 负样本固定使用对应折的 new_negative_pairs.txt 文件
########################################
if __name__ == "__main__":
    # 以 RPI1807 数据集为例，请根据实际情况修改路径
    ds_feature_folder = "/Users/ggcl7/Desktop/RPI审稿意见新结果/对比之前方法/RPI-capsuleGAN/merge"
    data_folder = "/Users/ggcl7/Desktop/RPI审稿意见新结果/分层评估对比之前方法/merge"
    #     # 从单一 Pairs.txt 文件中提取所有正样本
    # pairs_file = os.path.join("/Users/ggcl7/Desktop/RPI审稿意见新结果/分层评估对比之前方法/RPI1807", "Pairs.txt")
    # pos_pairs_all, _ = parse_pairs(pairs_file)  # 这里只取正样本

        # 读取保存好的蛋白质和 RNA 特征（CSV 文件中包含 ID 列）
    protein_features = pd.read_csv(os.path.join(ds_feature_folder, "protein_features.csv"))
    rna_features = pd.read_csv(os.path.join(ds_feature_folder, "rna_features.csv"))

    # 对于每一折，正样本和负样本均固定使用各折对应的文件
    all_fold_metrics = []
    for fold in range(1, 6):
        print(f"\n正在处理第 {fold} 折 ...")
        fold_dir = os.path.join(data_folder, f"fold{fold}")
        # 正样本固定
        train_pos_file = os.path.join(fold_dir, "train", "train_pairs.txt")
        test_pos_file = os.path.join(fold_dir, "test", "test_pairs.txt")
        # 负样本固定（均取自当前折对应目录）
        train_neg_file = os.path.join(os.path.dirname(train_pos_file), "new_negative_pairs.txt")
        test_neg_file = os.path.join(os.path.dirname(test_pos_file), "new_negative_pairs.txt")

        train_pos_pairs = load_positive_pairs(train_pos_file)
        test_pos_pairs = load_positive_pairs(test_pos_file)
        train_neg_pairs = load_positive_pairs(train_neg_file)
        test_neg_pairs = load_positive_pairs(test_neg_file)

        print(f"Fold {fold} 训练正样本数: {len(train_pos_pairs)}, 训练负样本数: {len(train_neg_pairs)}")
        print(f"Fold {fold} 测试正样本数: {len(test_pos_pairs)}, 测试负样本数: {len(test_neg_pairs)}")

        X_train, y_train = extract_features_from_pairs(protein_features, rna_features, train_pos_pairs, train_neg_pairs)
        X_test, y_test = extract_features_from_pairs(protein_features, rna_features, test_pos_pairs, test_neg_pairs)

        best_model, base_models = train_and_save_best_models(X_train, y_train, X_test, y_test)
        base_test_preds = get_base_predictions(base_models, X_test)
        y_pred = best_model.predict(base_test_preds)
        try:
            y_scores = best_model.decision_function(base_test_preds)
        except Exception:
            y_scores = best_model.predict_proba(base_test_preds)[:, 1]
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
    print("\n五折交叉验证结果：")
    for m in metrics_names:
        print(f"{m}: mean = {summary[m]['mean']:.3f}, std = {summary[m]['std']:.3f}")
