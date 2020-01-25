# BiDaF Baseline

This project is part of the AI laboratory of the Leibniz university Hanover (WiSe 19/20).

## Task
Given a question and a set of passages, rank them by relevance to the
question using the **BiDaF** model, for the following data sets:

- [MS MARCO Passage Retrieval](http://www.msmarco.org/dataset.aspx)
- [FiQA Task 2](https://sites.google.com/view/fiqa)
- [InsuranceQA V2](https://github.com/shuzi/insuranceQA)
- [ANTIQUE](https://ciir.cs.umass.edu/downloads/Antique/)

The resulting data wil be used as a baseline in further studies. In addition, it is desirable that the model can easily be extended by further data sets.

## Requirements
This code was run with Python 3.7.3 and:

- tqdm==4.40.2
- h5py==2.10.0
- numpy==1.18.1
- torch==1.3.1
- yaml==0.2.2
- nltk==3.4.4

You also have to download the Global Vectors for Word Representation since they're used in the Char Embedding by the Bidaf model:

- [glove.840B.300d.txt](https://nlp.stanford.edu/data/glove.840B.300d.zip)

## Preprocessing
You need to preprocess the dataset you want to use before you can start the training or evaluation. Simply download the datset and start the preprocess script to preprocess the data. We're using a Framework provided by the L3S which requires specific arguments for the different dataset.

```
usage: preprocess.py [-h] [-vs VOCAB_SIZE] [-n NUM_NEG_EXAMPLES]
                     SAVE {fiqa,msmarco,insrqa,wpqa} ...

positional arguments:
  SAVE                  Where to save the results
  {fiqa,msmarco,insrqa,antique}
                        Choose a dataset

optional arguments:
  -h, --help            show this help message and exit
  -vs VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        Vocabulary size
  -n NUM_NEG_EXAMPLES, --num_neg_examples NUM_NEG_EXAMPLES
                        Number of negative examples to sample
```

Examples for the different datasets:

MS MARCO Passage Retrieval:

  `path/to/preprocess.py msmarco path/to/msmarco_dataset`

FiQA:

  `path/to/preprocess.py fiqa path/to/fiqa_dataset path/to/BiDaF/qa_utils/splits/fiqa_split.pkl`

InsuranceQA:

  `path/to/preprocess.py insrqa path/to/insrqa_dataset --examples_per_query [100, 500, 1000, 1500]`

ANTIQUE:

  `path/to/preprocess.py antique path/to/antique_dataset path/to/BiDaF/qa_utils/splits/antique_split.pkl`

## Training
For the training you need use the preprocessed files. You can start the training by running the training script via:

```
usage: training.py [-h] [--force_restart] [--word_rep WORD_REP] [--cuda CUDA]
                   [--use_covariance] [--seed SEED]
                   exp_folder data dest

positional arguments:
  exp_folder           Experiment folder
  data                 Training data
  dest                 Destination folder

optional arguments:
  -h, --help           show this help message and exit
  --force_restart      Force restart of experiment: will ignore checkpoints
  --word_rep WORD_REP  Text file containing pre-trained word representations.
  --cuda CUDA          Use GPU if possible
  --use_covariance     Do not assume diagonal covariance matrix when
                       generating random word representations.
  --seed SEED          Seed for Pytorch
```

Example:

```
path/to/training.py path/to/Bidaf/experiment path/to/BiDaF/data path/to/Bidaf/results --word_rep path/to/glove.840B.300d.txt
```

You can change the configuration of the Bidaf model (e.g. number of epochs, batch size) by changing the `config.yaml` inside the experiment folder.

## Prediction
The evaluation uses the preprocessed files as well. You can start the evaluating of the trained model by running the prediction script via:

```
usage: prediction.py [-h] [--mrr_k MRR_K] [--word_rep WORD_REP]
                     [--batch_size BATCH_SIZE] [--cuda CUDA]
                     [--use_covariance] [--seed SEED] [--multi_gpu MULTI_GPU]
                     exp_folder data dest

positional arguments:
  exp_folder            Experiment folder
  data                  Prediction data
  dest                  Write predictions in

optional arguments:
  -h, --help            show this help message and exit
  --mrr_k MRR_K         Compute MRR@k
  --word_rep WORD_REP   Text file containing pre-trained word representations.
  --batch_size BATCH_SIZE
                        Batch size to use
  --cuda CUDA           Use GPU if possible
  --use_covariance      Do not assume diagonal covariance matrix when
                        generating random word representations.
  --seed SEED           Seed for Pytorch
  --multi_gpu MULTI_GPU
                        Use multiple GPUs for evaluation
```

Example:

```
path/to/prediction.py path/to/Bidaf/experiment path/to/BiDaF/data path/to/Bidaf/results --word_rep path/to/glove.840B.300d.txt
```

The result will be written as a csv file in the `results` folder, containing the information about MRR@K and MAP.

## Results

| Dataset     | Dev MAP | Dev MRR@10 | Test MAP | Test MRR@10 |
|:------------|--------:| ----------:|---------:|------------:|
| MS MARCO    | tba     | tba        | tba      | tba         |
| FiQA        | 0.5774  | 0.7666     | 0.5761   | 0.7524      |
| InsuranceQA | tba     | tba        | tba      | tba         |
| ANTIQUE     | 0.4641  | 0.7318     | 0.5793   | 0.8251      |

## Objectives

- Implement the BiDAF model
- Modify the output layer to produce a score of a passage w.r.t. a query
- Implement ranking metrics (MRR, MAP)
- Good documentation of the code
- (possibility to extend the project with other datasets)
