# TomNet: Few-Shot Learning for Function Call Graph Analysis
A comprehensive framework for few-shot learning on Function Call Graphs (FCGs), specifically designed for malware detection and classification tasks.

## Features

- Node Embedding: Word2Vec-based opcode embedding for graph nodes
Flexible Data Handling: Supports multiple CPU architectures and reverse engineering tools
- Graph Neural Network Support: GCN, GraphSAGE, GIN, GAT architectures
- Label Propagation Method: Advanced graph-based semi-supervised learning approach
- Open-Set Recognition: Support for open-set scenarios with AUROC evaluation

## Project Structure

```
TomNet/
├── config/
│   └── config.json              # Configuration file
├── preprocessing/
│   └── genGpickle.py           # FCG preprocessing utilities
├── dataset/                     # Dataset storage (to be created)
├── embeddings/                  # Node embeddings storage (auto-generated)
├── checkpoints/                 # Model checkpoints (auto-generated)
├── RunTraining.py              # Training script
├── RunEval.py                  # Evaluation script
├── loadDataset.py              # Dataset loading utilities
├── fcgVectorize.py             # Node embedding generation
├── trainModule.py              # Training and testing modules
├── models.py                   # Graph neural network models
├── loss.py                     # Loss functions for few-shot learning
├── dataset.py                  # Custom samplers for few-shot learning
├── train_utils.py              # Training utilities
└── utils.py                    # General utilities
```

## Installation
### Requirements

```
pip install torch torch-geometric
pip install pandas numpy scikit-learn
pip install networkx gensim
pip install tqdm
```

### Dependencies

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NetworkX
- Gensim (for Word2Vec)
- Pandas, NumPy, Scikit-learn

## Quick Start

1. Data Preparation
**Dataset Structure**
TomNet requires a specific directory structure for FCG data and CSV metadata files:
```
dataset/
├── raw_csv/                                    # CSV metadata files
│   ├── malware_diec_ghidra_x86_64_fcg_dataset.csv
│   └── malware_diec_ghidra_x86_64_fcg_openset_dataset.csv
├── data_ghidra_fcg/                           # Main FCG dataset name
│   └── {CPU}/                                 # CPU architecture (e.g., Advanced Micro Devices X86-64)
│       └── {family}/                          # Malware family name (e.g., adore)
│           ├── sample1.gpickle               # FCG files in NetworkX format
│           ├── sample2.gpickle
│           └── ...
├── data_ghidra_fcg_openset/                   # Open-set FCG dataset name
│   └── {CPU}/                                 # Same structure as main dataset
│       └── {family}/
│           ├── openset_sample1.gpickle
│           └── ...
├── split/                                     # Data split files (auto-generated)
│   ├── train_{datasetName}.txt
│   ├── test_{datasetName}.txt
│   └── val_{datasetName}.txt
└── embeddings/                                # Node embeddings (auto-generated)
    └── {datasetName}/
        └── word2vec/
            ├── opcode2vec.model
            └── ...
```

**CSV File Format**

Place your FCG dataset CSV files in the `dataset/raw_csv/` directory. The CSV should contain columns:

- `file_name`: Name of the binary file
- `family`: Malware family label
- `CPU`: CPU architecture (e.g., x86_64)

For open-set scenarios, prepare an additional openset dataset CSV.

**FCG Files**
Each FCG file should be:

Format: NetworkX graph saved as .gpickle
Location: dataset/{dataset_name}/{CPU}/{family}/{file_name}.gpickle
Node attributes: Each node should have an `'x'` attribute containing opcode sequences

Example FCG node structure:
```
# Node attributes in NetworkX graph
fcg.nodes[node_id]['x'] = ['push', 'mov', 'call', 'ret']  # Opcode sequence
```

2. Configuration

Edit `config/config.json` to customize your experiment:

```
{
    "dataset": {
        "pack_filter": "diec",
        "cpu_arch": "x86_64", 
        "reverse_tool": "ghidra",
        "raw": "your_dataset.csv",
        "openset": true,
        "openset_raw": "your_openset_dataset.csv"
    },
    "settings": {
        "name": "5way_5shot_LabelPropagation_gcn",
        "model": {
            "model_name": "GCN",
            "input_size": 128,
            "hidden_size": 256,
            "output_size": 128,
            "num_layers": 3
        },
        "few_shot": {
            "method": "LabelPropagation",
            "parameters": {
                "relation_layer": 2,
                "relation_model": "GCN",
                "dim_in": 128,
                "dim_hidden": 64,
                "dim_out": 32,
                "rn": 300,
                "alpha": 0.7,
                "k": 20
            }
        }
    }
}
```

3. Training

```
python RunTraining.py --config config/config.json
```

4. Evaluation

```
python RunEval.py --config path/to/saved/config.json
```

## Data Preprocessing
If you have raw FCG data from reverse engineering tools (like Ghidra), use the preprocessing script:
```
# Edit the paths in preprocessing/genGpickle.py
python preprocessing/genGpickle.py
```
The script expects:

Input: Raw FCG files (.dot and .json format)
Output: Processed .gpickle files with opcode attributes


## Label Propagation Method
Label Propagation is the primary few-shot learning method in TomNet, which leverages graph structure for semi-supervised learning. The method works by:

1. Feature Extraction: Extract node embeddings using GNN encoder
2. Graph Construction: Build similarity graph between support and query samples
3. Label Propagation: Propagate labels from support to query samples through the graph
4. Classification: Make predictions based on propagated labels

### Algorithm Overview
The Label Propagation algorithm follows these steps:

1. Embedding: Use GNN encoder to get graph-level embeddings
2. Similarity Matrix: Compute similarity matrix W between all samples
3. Normalization: Normalize similarity matrix to get transition matrix S
4. Propagation: Solve F = (I - αS)^(-1)Y where Y contains support labels
5. Prediction: Use propagated labels F for classification

### Parameters Explanation
The Label Propagation method uses several key parameters in the `few_shot.parameters` section:
**Graph Relation Network Parameters**

- `relation_layer` (int, default: 2): Number of layers in the relation network

    - Controls the depth of the GNN used for computing similarity
    - More layers can capture more complex relationships but may cause over-smoothing


- `relation_model` (str, default: "GCN"): Type of GNN for relation learning

    - Options: "GCN", "GraphSAGE", "GIN", "GAT"
    - Different architectures have different inductive biases


- `dim_in` (int, default: 128): Input dimension for relation network

    - Should match the output dimension of the main encoder
    - Typically set to `model.output_size`


- `dim_hidden` (int, default: 64): Hidden dimension in relation network

    - Controls the capacity of the relation network
    - Balance between expressiveness and overfitting


- `dim_out` (int, default: 32): Output dimension of relation network

    - Final embedding dimension before similarity computation
    - Smaller values may improve generalization

### Label Propagation Parameters

- `rn` (int, default: 300): Relation network configuration

    - `30`: Both sigma and alpha are learnable parameters
    - `300`: Sigma is learnable, alpha is fixed
    - Controls whether the propagation strength is adaptive


- `alpha` (float, default: 0.7): Label propagation strength

    - Range: [0, 1]
    - Higher values give more weight to graph structure
    - Lower values rely more on initial labels


- `k` (int, default: 20): Number of nearest neighbors to keep

    - Sparsifies the similarity graph
    - `k=0` means dense graph (all connections)
    - Higher k preserves more graph structure but increases computation

## Open-Set Recognition

TomNet supports open-set scenarios where test samples may belong to classes not seen during training:

```
"openset": {
    "train": {
        "use": true,
        "m_samples": 20,
        "class_per_iter": 5,
        "loss_weight": 0.5
    },
    "test": {
        "use": true,
        "m_samples": 50,
        "class_per_iter": 5
    }
}
```
The framework uses entropy-based losses and AUROC metrics for open-set evaluation with Label Propagation.

## Results and Logging
Training results are saved to:

- Model checkpoints: `checkpoints/[experiment_name]/`
- Training logs: `log.txt`
- Evaluation results: `evalLog.csv` or `evalLog_openset.csv`

## Example Usage
### Closed-Set 5-way 5-shot Classification
```
# Configure for closed-set scenario
python RunTraining.py --config config/closedset_config.json

# Evaluate
python RunEval.py --config checkpoints/experiment/config.json
```
### Open-Set Recognition
```
# Configure for open-set scenario  
python RunTraining.py --config config/openset_config.json

# Evaluate with open-set testing
python RunEval.py --config checkpoints/experiment/config.json
```

## Citation

## License

## Related Work
Please also consider citing the foundational works that inspired this framework:
```
@article{liu2018learning,
  title={Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning},
  author={Liu, Yanbin and Lee, Juho and Park, Minseop and Kim, Saehoon and Yang, Eunho and Hwang, Sung Ju and Yang, Yi},
  journal={arXiv preprint arXiv:1805.10002},
  year={2018}
}

@inproceedings{liu2020few,
  title={Few-Shot Open-Set Recognition Using Meta-Learning},
  author={Liu, Bo and Kang, Hao and Li, Haoxiang and Hua, Gang and Vasconcelos, Nuno},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8798--8807},
  year={2020}
}
```
## Acknowledgments
- PyTorch Geometric team for the excellent graph learning library
- Liu et al. for [Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning](https://arxiv.org/abs/1805.10002)
- Liu et al. for [Few-Shot Open-Set Recognition Using Meta-Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Few-Shot_Open-Set_Recognition_Using_Meta-Learning_CVPR_2020_paper.html)
