# PASER
## Requriements
You can install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
In this framework, we use the widely-adopted IEMOCAP dataset for training and evaluation. The dataset can be downloaded from [this link](https://sail.usc.edu/iemocap/). The specific process is as follows:

1. Download the IEMOCAP dataset to your directory and extract it.

2. Extract the audio signals, emotion labels, and transcribed text from the dataset. Then, convert the text sequences into phoneme sequences using the cmudict from the nltk library. The specific steps are as follows:


```bash
cd dataset/IEMOCAP
bash preprocess.sh --dataset_dir $path to directory of IEMOCAP$
cd ../..
```

The above data is packaged into `iemocap4char.pkl`.
## Train and Evalution

To facilitate running the code, we have packaged the execution commands into `run_model.sh`. The specific steps are as follows:

```bash
bash run_model.sh
```

`run_model.sh` contains important hyperparameters:

- `--train_strategy`: 
  - 0 indicates the use of fixed task weights
  - 4 represents the use of dynamic task priority
  - Other values(1,2,3) correspond to strategies that can be referenced in the code

- `--ablation_level`: 
  - -1 means no ablation of any module
  - 0 indicates complete ablation (direct fine-tuning)
  - 1 represents `w/o phoneme decoder`
  - 2 represents `w/o SE module`

- `--weight_phoneme`: The weight corresponding to the phoneme recognition task when using a fixed weight strategy

Additionally, the authors experimented with various feature fusion strategies (`--fuse_mode`) and pooling strategies (`--mode`). Using the default values of `concat` and `mean` can achieve good results.
