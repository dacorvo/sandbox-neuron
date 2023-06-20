# Testing optimum-neuron

## Prerequisites

At the time of creation of this repository, this requires:
- the latest HF AMI,
- optimum 1.8.8,
- optimum-neuron 0.0.5dev

```
pip install -r requirements.txt
```

## Inference CLI

The inference CLI can perform an inference on either a model from the Hub or from a local directory.

```
usage: inference.py [-h]
                    [--model MODEL]
                    [--dataset DATASET]
                    [--num-samples NUM_SAMPLES]
                    [--metric METRIC]
                    [--input-column INPUT_COLUMN]
                    [--batch-size BATCH_SIZE]
                    [--seq-length SEQ_LENGTH]
                    [--save-dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The model to use for inference: can be a model ID, or the path to a local directory.
  --dataset DATASET     The dataset on which the model should be evaluated.
  --num-samples NUM_SAMPLES
                        The number of samples to use for evaluation.
  --metric METRIC       The metric used to evaluate the model.
  --input-column INPUT_COLUMN
                        The name of the input column in the selecetd dataset.
  --batch-size BATCH_SIZE
                        The static batch size of the Neuron model.
  --seq-length SEQ_LENGTH
                        The static sequence length of the Neuron model.
  --save-dir SAVE_DIR   The path to save the Neuron model.
  ```

  >Note: when selecting a model from the Hub, it will be converted to a Neuron model. If a save directory is specified, it
  will be saved there. This is strictly equivalent to exporting the model using the `optimum-cli export neuron` command.


## Test-classification results


### sst2 dataset

Bert

```
$ python inference.py --model kowsiknd/bert-base-uncased-sst2  \
                      --dataset sst2                           \
                      --input-column sentence                  \
                      --metric accuracy                        \
                      --batch-size 1                           \
                      --seq-length 128
{'accuracy': 0.8692660550458715, 'total_time_in_seconds': 2.378855768001813, 'samples_per_second': 366.56278692022636, 'latency_in_seconds': 0.002728045605506666}
```

DistilBert

```
$ python inference.py --model distilbert-base-uncased-finetuned-sst-2-english  \
                      --dataset sst2                                           \
                      --input-column sentence                                  \
                      --metric accuracy                                        \
                      --batch-size 1                                           \
                      --seq-length 128
{'accuracy': 0.9105504587155964, 'total_time_in_seconds': 1.6696950630030187, 'samples_per_second': 522.2510500999329, 'latency_in_seconds': 0.0019147879162878653}
```

RoBerta

```
$ python inference.py --model Ibrahim-Alam/finetuning-roberta-base-on-sst2 \
                      --dataset sst2                                       \
                      --input-column sentence                              \
                      --metric accuracy                                    \
                      --batch-size 1                                       \
                      --seq-length 128
{'accuracy': 0.9415137614678899, 'total_time_in_seconds': 2.2805290400028753, 'samples_per_second': 382.3674176930896, 'latency_in_seconds': 0.0026152855963335725}
```