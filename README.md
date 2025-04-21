# Graph-RPI: Reproduction of Experimental Results

This repository provides the full codebase and data required to reproduce all results presented in our paper on **Graph-RPI**, a graph-based model for RNA–protein interaction (RPI) prediction. It includes the implementation of the proposed method, multiple baseline methods, ablation studies, interpretability analysis, and evaluation under five-fold cross-validation. The goal is to ensure the reproducibility and transparency of every analysis in the article.

The repository contains the following components:

- **Ablation experiments**: Includes scripts to evaluate the effect of different design choices such as feature fusion strategies (`different feature fusion.py`), GNN variants (`different GNN layers.py`), negative sampling strategies (`different negative sampling strategies.py`), masking ratio and alpha hyperparameters (`mask ratio and alpha value settings.py`), and supervised vs. self-supervised regimes (`supervised and self-supervised regimes.py`).

- **Comparison methods**: Provides implementations or wrappers for multiple RPI or LPI prediction baselines, including `LPI-CNNCP`, `LPI-SKMSC`, `RPI-CapsGAN`, `RPI-MDLStack`, and `RPISeq`. These methods are evaluated under the same datasets and cross-validation settings for fair comparison.

- **Five-fold cross-validation data**: Contains predefined fold partitions for all datasets used in the paper. These folds are used consistently across all models to ensure a fair and replicable evaluation.

- **Model interpretability analysis**: Includes scripts for SHAP (Shapley Additive Explanations) analysis based on the Captum library. These scripts help identify which RNA and protein features are most influential in the Graph-RPI predictions.

- **Similarity**: Contains clustering code and similarity matrices used to construct more stringent data splits and validate model generalization beyond sequence similarity.

- **train_test data**: Stores the original benchmark RPI datasets used for training and testing.

## 🔧 Installation instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourname/Graph-RPI-All-Experiments.git
cd Graph-RPI-All-Experiments
```
2. **Set up the Python environment**
```bash
conda create -n graphrpi python=3.10
conda activate graphrpi
pip install -r requirements.txt
```
3. **ESM-2 Language model embeddings**
```bash
https://huggingface.co/facebook/esm2_t6_8M_UR50D
```

## 🚀 Running the Graph-RPI Model
To train the Graph-RPI model, navigate to the model folder and execute the main script:

```bash
cd Graph-RPI-Model
python main.py
```

## 📄 Citations
If you use this repository in your work, please cite these papers:
```bash
@article{sun2024lpi,
  title={LPI-SKMSC: Predicting LncRNA--Protein Interactions with Segmented k-mer Frequencies and Multi-space Clustering},
  author={Sun, Dian-Zheng and Sun, Zhan-Li and Liu, Mengya and Yong, Shuang-Hao},
  journal={Interdisciplinary Sciences: Computational Life Sciences},
  volume={16},
  number={2},
  pages={378--391},
  year={2024},
  publisher={Springer}
}
@article{wang2023rpi,
  title={RPI-CapsuleGAN: Predicting RNA-protein interactions through an interpretable generative adversarial capsule network},
  author={Wang, Yifei and Wang, Xue and Chen, Cheng and Gao, Hongli and Salhi, Adil and Gao, Xin and Yu, Bin},
  journal={Pattern Recognition},
  volume={141},
  pages={109626},
  year={2023},
  publisher={Elsevier}
}
@article{yu2022rpi,
  title={RPI-MDLStack: Predicting RNA--protein interactions through deep learning with stacking strategy and LASSO},
  author={Yu, Bin and Wang, Xue and Zhang, Yaqun and Gao, Hongli and Wang, Yifei and Liu, Yushuang and Gao, Xin},
  journal={Applied Soft Computing},
  volume={120},
  pages={108676},
  year={2022},
  publisher={Elsevier}
}
@article{zhang2020lpi,
  title={LPI-CNNCP: Prediction of lncRNA-protein interactions by using convolutional neural network with the copy-padding trick},
  author={Zhang, Shao-Wu and Zhang, Xi-Xi and Fan, Xiao-Nan and Li, Wei-Na},
  journal={Analytical biochemistry},
  volume={601},
  pages={113767},
  year={2020},
  publisher={Elsevier}
}
@article{muppirala2011predicting,
  title={Predicting RNA-protein interactions using only sequence information},
  author={Muppirala, Usha K and Honavar, Vasant G and Dobbs, Drena},
  journal={BMC bioinformatics},
  volume={12},
  pages={1--11},
  year={2011},
  publisher={Springer}
}
```
