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
                                      Hub                Neuron
accuracy                                    0.8693              0.8693
total_time_in_seconds                      68.1093              2.2840
samples_per_second                         12.8029            381.7842
latency_in_seconds                          0.0781              0.0026
```

`model=distilbert-base-uncased-finetuned-sst-2-english`
```
                                      Hub                Neuron
accuracy                                    0.9106              0.9106
total_time_in_seconds                      34.5766              1.7844
samples_per_second                         25.2194            488.6926
latency_in_seconds                          0.0397              0.0020
```

`model=Ibrahim-Alam/finetuning-roberta-base-on-sst2`
```
                                      Hub                Neuron
accuracy                                    0.9415              0.9415
total_time_in_seconds                      67.5452              2.2621
samples_per_second                         12.9099            385.4777
latency_in_seconds                          0.0775              0.0026
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
                                      Hub                Neuron
accuracy                                    0.8270              0.8270
total_time_in_seconds                      78.4664              3.2580
samples_per_second                         12.7443            306.9348
latency_in_seconds                          0.0785              0.0033
```

`model=lvwerra/distilbert-imdb`
```
                                      Hub                Neuron
accuracy                                    0.8670              0.8670
total_time_in_seconds                      40.2383              2.3156
samples_per_second                         24.8520            431.8515
latency_in_seconds                          0.0402              0.0023
```

`model=aychang/roberta-base-imdb`
```
                                      Hub                Neuron
accuracy                                    0.8440              0.8440
total_time_in_seconds                      80.0236              3.1624
samples_per_second                         12.4963            316.2131
latency_in_seconds                          0.0800              0.0032
```