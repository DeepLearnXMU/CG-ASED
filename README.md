# ASED

Codes for paper "An AST Structure Enhanced Decoder for Code Generation".

## System Architecture

The source code is developed upon TranX,
for technical details please refer to [ACL '18 paper](https://arxiv.org/abs/1806.07832) and [EMNLP '18 demo paper](https://arxiv.org/abs/1810.02720). 

**File Structure** is mainly composed of two components: 

* `asdl` defines a general-purpose transition system based on the ASDL formalism, and its instantiations in different programming languages and datasets. The transition system defines how an AST is constructed using a sequence of actions. This package can be used as a standalone library independent of tranX.

* `model` contains the neural network implementation of the transition system defined in `asdl`, which computes action probabilities using neural networks.See Section 2.3 of the technical report for details.

Here is a detailed map of the file strcuture:
```bash
├── asdl (grammar-based transition system)
├── datasets (dataset specific code like data preprocessing/evaluation/etc.)
├── model (PyTorch implementation of neural nets)
├── server (interactive Web server)
├── components (helper functions and classes like vocabulary)
```

## Usage
### Step 1: Download the datasets
```bash
cd ASED
bash ./pull_data.sh # get datasets from the TranX
```

### Step 2: Preprocess the data
```bash
python datasets/django/dataset.py # preprocess the django dataset 
```

### Step 3: Train the model
```bash
sh scripts/django/train.sh # train model on the django dataset 
```
At last, it will show the accuracy on test dataset.

### Evaluation Results
Here is a list of performance results on four datasets using pretrained models in `pretrained_models`

| Dataset | Results      | Metric             |
| ------- | ------------ | ------------------ |
| Django  | 79.72        | Accuracy           |
| CoNaLa  | 26.32        | Corpus BLEU        |
| ATIS    | 89.06         | Accuracy           |
| GEO     | 91.07         | Accuracy           |

You can run `sh scripts/<lang>/test.sh` to get these results.

### Conda Environments

Please note that 
Django dataset only support Python 2.7.
The main example conda environment (`config/env/tranx.yml`) supports Python 3, and
environment (`config/env/tranx-py2.yml`) supports Python 2.
You can export the enviroments using the following command:

```bash
conda env create -f config/env/(tranx.yml,tranx-py2.yml)
```


