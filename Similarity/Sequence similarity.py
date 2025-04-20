import os
import csv
from Bio.Align import PairwiseAligner, substitution_matrices



def is_valid_seq(seq, valid_chars):
    for ch in seq:
        if ch not in valid_chars:
            return False
    return True


def read_fasta_to_dict(file_path, valid_chars):
    seq_dict = {}
    header = None
    sequence = ""
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    if is_valid_seq(sequence, valid_chars):
                        seq_dict[header] = sequence
                header = line[1:].strip()
                sequence = ""
            else:
                sequence += line
        if header is not None:
            if is_valid_seq(sequence, valid_chars):
                seq_dict[header] = sequence
    return seq_dict



def parse_pairs(file_path):
    positive_pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            protein_id, rna_id, label = line.strip().split('\t')
            if int(label) == 1:
                positive_pairs.append((protein_id, rna_id))
    return positive_pairs



def compute_rna_similarity(seq1, seq2):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    best_score = aligner.score(seq1, seq2)
    return best_score


def compute_protein_similarity(seq1, seq2):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    blosum62 = substitution_matrices.load("BLOSUM62")
    aligner.substitution_matrix = blosum62
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    best_score = aligner.score(seq1, seq2)
    return best_score


def compute_pair_similarity(pair1, pair2, protein_dict, rna_dict, alpha=0.5):
    protein1, rna1 = pair1
    protein2, rna2 = pair2

    seq_protein1 = protein_dict.get(protein1, "")
    seq_protein2 = protein_dict.get(protein2, "")
    seq_rna1 = rna_dict.get(rna1, "")
    seq_rna2 = rna_dict.get(rna2, "")

    sim_protein = compute_protein_similarity(seq_protein1, seq_protein2) if seq_protein1 and seq_protein2 else 0
    sim_rna = compute_rna_similarity(seq_rna1, seq_rna2) if seq_rna1 and seq_rna2 else 0

    combined_score = alpha * sim_rna + (1 - alpha) * sim_protein
    return combined_score


def main():

    protein_file = "Protein.txt"
    rna_file = "RNA.txt"
    pairs_file = "Pairs.txt"

    valid_protein_chars = set("ARNDCQEGHILKMFPSTWYV")

    valid_rna_chars = set("ACGU")

    protein_dict = read_fasta_to_dict(protein_file, valid_protein_chars)
    rna_dict = read_fasta_to_dict(rna_file, valid_rna_chars)


    pairs = parse_pairs(pairs_file)


    filtered_pairs = []
    for pair in pairs:
        protein_id, rna_id = pair
        if protein_id in protein_dict and rna_id in rna_dict:
            filtered_pairs.append(pair)
    pairs = filtered_pairs


    if not pairs:
        print("没有有效的蛋白-RNA pair可供计算。")
        return


    n = len(pairs)
    similarity_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            score = compute_pair_similarity(pairs[i], pairs[j], protein_dict, rna_dict, alpha=0.5)
            similarity_matrix[i][j] = score
            similarity_matrix[j][i] = score

    for row in similarity_matrix:
        print("\t".join("{:.2f}".format(x) for x in row))


    pair_labels = [f"{protein}|{rna}" for protein, rna in pairs]


    output_csv = "similarity_matrix.csv"
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        header_row = ["Pair_ID"] + pair_labels
        writer.writerow(header_row)

        for label, row in zip(pair_labels, similarity_matrix):
            writer.writerow([label] + row)



if __name__ == "__main__":
    main()
