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

```
$ model=<model>
$ python inference.py --dataset sst2             \
                      --input-column sentence    \
                      --metric accuracy          \
                      --batch-size 1             \
                      --seq-length 128           \
                      --model $model
```

`model=kowsiknd/bert-base-uncased-sst2`:
```
accuracy: 0.8693
total_time_in_seconds: 2.3734
samples_per_second: 367.4128
latency_in_seconds: 0.0027
```

`model=distilbert-base-uncased-finetuned-sst-2-english`
```
accuracy: 0.9106
total_time_in_seconds: 1.7009
samples_per_second: 512.6797
latency_in_seconds: 0.0020
```

`model=Ibrahim-Alam/finetuning-roberta-base-on-sst2`
```
accuracy: 0.9415
total_time_in_seconds: 2.2931
samples_per_second: 380.2731
latency_in_seconds: 0.0026
```

### IMDB dataset

```
$ model=<model>
$ python inference.py --dataset imdb      \
                      --metric accuracy   \
                      --batch-size 1      \
                      --seq-length 128    \
                      --num-samples 1000  \
                      --model $model
```

`model=fabriceyhc/bert-base-uncased-imdb`
```
accuracy: 0.8270
total_time_in_seconds: 3.2782
samples_per_second: 305.0423
latency_in_seconds: 0.0033
```

`model=lvwerra/distilbert-imdb`
```
accuracy: 0.8670
total_time_in_seconds: 2.4173
samples_per_second: 413.6885
latency_in_seconds: 0.0024
```

`model=aychang/roberta-base-imdb`
```
accuracy: 0.8440
total_time_in_seconds: 3.1583
samples_per_second: 316.6219
latency_in_seconds: 0.0032
```