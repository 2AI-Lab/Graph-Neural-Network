# Fake News Detection using Graph Neural Networks - GATs with Pytorch Geometric

This repository contains code for detecting fake news using graph neural networks (GNNs) with PyTorch Geometric (PyG). The code is based on the UPFD dataset, which consists of user posts from online forums. The code includes preprocessing and data loading functions, a GNN model for fake news detection, and evaluation functions for assessing the performance of the model.

## Requirements
The code in this repository is written in Python 3.7 and requires the following libraries:

* PyTorch
* PyTorch Geometric (PyG)
* Pandas
* Spacy

You can install these libraries using pip:

```
!pip install torch-geometric pandas spacy
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
!pip install -q git+https://github.com/rusty1s/pytorch_geometric.git
```

## Running the code
To run the code, clone the repository and navigate to the root directory:

```
git clone https://github.com/2AI-Lab/Graph-Neural-Network/tree/main/Case-Studies/Fraud-Detection.git
cd fake-news-detection-pyg
```

## Credits
The code in this repository is based on the UPFD dataset and the work described in the UPFD paper:

K. Dai, Y. Li, Y. Cui, Y. Wang, and X. Li, "UPFD: A Large-Scale User Post Fake News Detection Dataset and Benchmark," arXiv preprint arXiv:2104.12259 (2021).
The code also uses the Spacy Word2Vec model for preprocessing the text data.

## Disclaimer:
This code is provided for educational purposes only and is not intended for production use. The authors and contributors of this repository are not responsible for any errors or omissions in the code or for any damages resulting from its use. Please use at your own risk.
