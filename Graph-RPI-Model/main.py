from Model import *
from RNA_feature import generate_features_rna
from protein_feature import generate_features_protein


def build_edge_index(pairs):
    edge_list = []
    for p, r in pairs:
        if (p in protein_id_to_index) and (r in rna_id_to_index):
            rna_idx = rna_id_to_index[r]
            protein_idx = protein_id_to_index[p]
            edge_list.append([rna_idx, protein_idx + len(rna_ids)])
    if len(edge_list) == 0:
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




def train_and_evaluate(model, train_data, test_data, optimizer, num_epochs=3000, alpha=0.4):
    best_acc = 0
    best_state = None
    best_metrics = None
    for epoch in range(num_epochs):
        model.train()
        model.train_epoch(train_data, optimizer, alpha)
        model.eval()
        with torch.no_grad():
            z = model.encoder(test_data.x, test_data.edge_index)
            metrics = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
        if metrics[2] > best_acc:
            best_acc = metrics[2]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:03d}: Test ACC={metrics[2]:.4f}")
    return best_metrics, best_state



if __name__ == '__main__':

    dataset_name = "RPImerged"
    base_path = "Five-fold cross-validation data"
    data_path = os.path.join(base_path, dataset_name)
    folds_path = os.path.join(data_path, "folds")
    save_path = os.path.join(base_path, dataset_name)
    os.makedirs(save_path, exist_ok=True)


    protein_file = os.path.join(data_path, "Protein.txt")
    rna_file = os.path.join(data_path, "RNA.txt")
    pairs_file = os.path.join(data_path, "Pairs.txt")


    protein_sequences = read_protein_sequences_from_fasta(protein_file)

    rna_features = generate_features_rna(rna_file)
    print(f"Generated RNA features with shape: {rna_features.shape}")

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

    results = []
    for fold in range(1, 6):
        print(f"\n🔁 Fold {fold}")
        set_seed(2023 + fold)

        test_fold_file = os.path.join(folds_path, f"fold{fold}_pairs.txt")
        test_pos_pairs, _ = parse_pairs(test_fold_file)
        test_pairs = test_pos_pairs
        train_pairs = []
        for i in range(1, 6):
            if i != fold:
                fold_file = os.path.join(folds_path, f"fold{i}_pairs.txt")
                train_pos_pairs, _ = parse_pairs(fold_file)
                train_pairs.extend(train_pos_pairs)


        transform = T.RandomLinkSplit(
            num_val=0,
            num_test=0.2,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=True
        )
        train_data_rand, _, test_data_rand = transform(data_all)

        train_data_rand.pos_edge_label_index = build_edge_index(train_pairs)
        test_data_rand.pos_edge_label_index = build_edge_index(test_pairs)

        train_data_rand.neg_edge_label_index = filter_bipartite_negatives(train_data_rand.neg_edge_label_index,
                                                                          data_all.num_rna)
        test_data_rand.neg_edge_label_index = filter_bipartite_negatives(test_data_rand.neg_edge_label_index,
                                                                         data_all.num_rna)


        train_neg_file = os.path.join(base_path, dataset_name, f"fold{fold}", "train", "negative_pairs.txt")
        neg_pairs_train = parse_pairs_only(train_neg_file)
        train_data_rand.neg_edge_label_index = build_edge_index(neg_pairs_train)
        num_pos_train = train_data_rand.pos_edge_label_index.size(1)
        train_neg = train_data_rand.neg_edge_label_index
        if train_neg.size(1) < num_pos_train:
            indices = torch.randint(0, train_neg.size(1), (num_pos_train,), device=train_neg.device)
            train_data_rand.neg_edge_label_index = train_neg[:, indices]
        elif train_neg.size(1) > num_pos_train:
            indices = torch.randperm(train_neg.size(1))[:num_pos_train]
            train_data_rand.neg_edge_label_index = train_neg[:, indices]


        test_neg_file = os.path.join(base_path, dataset_name, f"fold{fold}", "test", "negative_pairs.txt")
        neg_pairs_test = parse_pairs_only(test_neg_file)
        test_data_rand.neg_edge_label_index = build_edge_index(neg_pairs_test)
        num_pos_test = test_data_rand.pos_edge_label_index.size(1)
        test_neg = test_data_rand.neg_edge_label_index
        if test_neg.size(1) < num_pos_test:
            indices = torch.randint(0, test_neg.size(1), (num_pos_test,), device=test_neg.device)
            test_data_rand.neg_edge_label_index = test_neg[:, indices]
        elif test_neg.size(1) > num_pos_test:
            indices = torch.randperm(test_neg.size(1))[:num_pos_test]
            test_data_rand.neg_edge_label_index = test_neg[:, indices]

        train_data = train_data_rand
        test_data = test_data_rand


        encoder = GNNEncoder(in_channels=output_dim, hidden_channels=64, out_channels=128, heads=8)
        edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
        degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
        model = RPI(encoder, edge_decoder, degree_decoder, Mask(p=0.4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)


        metrics, best_state = train_and_evaluate(model, train_data, test_data, optimizer)
        results.append((dataset_name, fold, *metrics))

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            z_train = model.encoder(train_data.x, train_data.edge_index)
            train_metrics = model.test(z_train, train_data.pos_edge_label_index, train_data.neg_edge_label_index)
            z_test = model.encoder(test_data.x, test_data.edge_index)
            test_metrics = model.test(z_test, test_data.pos_edge_label_index, test_data.neg_edge_label_index)

        train_auc, train_ap, train_acc, train_sen, train_pre, train_spe, train_f1, train_mcc = train_metrics
        test_auc, test_ap, test_acc, test_sen, test_pre, test_spe, test_f1, test_mcc = test_metrics

        train_pos_count = train_data.pos_edge_label_index.size(1)
        train_neg_count = train_data.neg_edge_label_index.size(1)
        test_pos_count = test_data.pos_edge_label_index.size(1)
        test_neg_count = test_data.neg_edge_label_index.size(1)

        print(f"Fold {fold}:")
        print(f"  Train set: {train_pos_count} positive, {train_neg_count} negative samples")
        print(f"  Test set:  {test_pos_count} positive, {test_neg_count} negative samples")
        print(
            f"  Train metrics: ACC={train_acc:.4f}, SEN={train_sen:.4f}, SPE={train_spe:.4f}, MCC={train_mcc:.4f}, F1={train_f1:.4f}, Precision={train_pre:.4f}, AUC={train_auc:.4f}")
        print(
            f"  Test metrics:  ACC={test_acc:.4f}, SEN={test_sen:.4f}, SPE={test_spe:.4f}, MCC={test_mcc:.4f}, F1={test_f1:.4f}, Precision={test_pre:.4f}, AUC={test_auc:.4f}")

        fold_dir = os.path.join(save_path, f"fold{fold}")
        os.makedirs(os.path.join(fold_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "test"), exist_ok=True)

        torch.save(train_data.x, os.path.join(fold_dir, "train", "x.pt"))
        torch.save(train_data.edge_index, os.path.join(fold_dir, "train", "edge_index.pt"))
        torch.save(train_data.pos_edge_label_index, os.path.join(fold_dir, "train", "pos_edge_label_index.pt"))
        torch.save(train_data.neg_edge_label_index, os.path.join(fold_dir, "train", "neg_edge_label_index.pt"))

        torch.save(test_data.x, os.path.join(fold_dir, "test", "x.pt"))
        torch.save(test_data.edge_index, os.path.join(fold_dir, "test", "edge_index.pt"))
        torch.save(test_data.pos_edge_label_index, os.path.join(fold_dir, "test", "pos_edge_label_index.pt"))
        torch.save(test_data.neg_edge_label_index, os.path.join(fold_dir, "test", "neg_edge_label_index.pt"))

        train_pairs_file = os.path.join(fold_dir, "train", "train_pairs.txt")
        save_pairs(train_data.pos_edge_label_index, train_pairs_file)
        test_pairs_file = os.path.join(fold_dir, "test", "test_pairs.txt")
        with open(test_pairs_file, "w") as f:
            for p, r in test_pairs:
                f.write(f"{p}\t{r}\n")

        torch.save(best_state, os.path.join(fold_dir, f"best_model_fold{fold}.pth"))

    import pandas as pd

    results_df = pd.DataFrame(results,
                              columns=["Dataset", "Fold", "AUC", "AP", "ACC", "SEN", "PRE", "SPE", "F1", "MCC"])
    results_df.to_csv(os.path.join(save_path, f"results_{dataset_name}.csv"), index=False)
    results_df.groupby("Dataset").mean().reset_index().to_csv(
        os.path.join(save_path, f"avg_results_{dataset_name}.csv"), index=False
    )


