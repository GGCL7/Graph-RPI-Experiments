
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import iFeatureOmegaCLI  # 确保已经安装了这个库
import re
import math

def DDE(fastas, **kw):
    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H',
          'I', 'K', 'L', 'M', 'N', 'P', 'Q',
          'R', 'S', 'T', 'V', 'W', 'Y']

    myCodons = {
        'A': 4,
        'C': 2,
        'D': 2,
        'E': 2,
        'F': 2,
        'G': 4,
        'H': 2,
        'I': 3,
        'K': 2,
        'L': 6,
        'M': 1,
        'N': 2,
        'P': 4,
        'Q': 2,
        'R': 6,
        'S': 6,
        'T': 4,
        'V': 4,
        'W': 1,
        'Y': 2
    }

    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#'] + diPeptides
    encodings.append(header)

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

    AADict = {aa: i for i, aa in enumerate(AA)}

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            if sequence[j] in AADict and sequence[j+1] in AADict:
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] += 1
        if sum(tmpCode) != 0:
            tmpCode = [x / sum(tmpCode) for x in tmpCode]

        myTV = []
        for j in range(len(myTM)):
            myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

        for j in range(len(tmpCode)):
            tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

        code = code + tmpCode
        encodings.append(code)
    return encodings

def feature_DDE(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    fasta_list = np.array(f.readlines())
    aa_feature_list = []

    for flag in range(0, len(fasta_list), 2):
        if flag + 1 < len(fasta_list):
            fasta_str = [[fasta_list[flag].strip('\n').strip(), fasta_list[flag + 1].strip('\n').strip()]]

            dpc_output = DDE(fasta_str)
            dpc_output[1].remove(dpc_output[1][0])
            dpc_feature = dpc_output[1][:]
            aa_feature_list.append(dpc_feature)
        else:
            print(f"Warning: Missing sequence for header {fasta_list[flag].strip()}")

    aa_feature_list = pd.DataFrame(aa_feature_list)
    coloumnname = [f'DDE{i+1}' for i in range(len(aa_feature_list.columns))]
    aa_feature_list.columns = coloumnname
    return aa_feature_list

def generate_features_protein(input_txt_path):
    CKSAAGP = iFeatureOmegaCLI.iProtein(input_txt_path)
    CKSAAGP.get_descriptor("CKSAAGP type 2")
    CKSAAGP.display_feature_types()

    AAC = iFeatureOmegaCLI.iProtein(input_txt_path)
    AAC.get_descriptor("AAC")

    PAAC = iFeatureOmegaCLI.iProtein(input_txt_path)
    PAAC.get_descriptor("PAAC")

    QSOrder = iFeatureOmegaCLI.iProtein(input_txt_path)
    QSOrder.get_descriptor("QSOrder")

    GTPC = iFeatureOmegaCLI.iProtein(input_txt_path)
    GTPC.get_descriptor("GTPC type 2")

    GDPC = iFeatureOmegaCLI.iProtein(input_txt_path)
    GDPC.get_descriptor("GDPC type 2")

    DistancePair = iFeatureOmegaCLI.iProtein(input_txt_path)
    DistancePair.get_descriptor("DistancePair")

    dde = feature_DDE(input_txt_path)


    AAC.encodings = AAC.encodings.reset_index(drop=True)
    PAAC.encodings = PAAC.encodings.reset_index(drop=True)
    DistancePair.encodings = DistancePair.encodings.reset_index(drop=True)
    CKSAAGP.encodings = CKSAAGP.encodings.reset_index(drop=True)
    GTPC.encodings = GTPC.encodings.reset_index(drop=True)
    GDPC.encodings = GDPC.encodings.reset_index(drop=True)
    QSOrder.encodings = QSOrder.encodings.reset_index(drop=True)
    dde = dde.reset_index(drop=True)


    result = pd.concat([AAC.encodings, PAAC.encodings, CKSAAGP.encodings, QSOrder.encodings, dde], axis=1)
    result.index = PAAC.encodings.index
    result = torch.tensor(result.iloc[:, :].values, dtype=torch.float32)
    return result.numpy()


