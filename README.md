# SACHA

The code implementation for SACHA: Soft Actor-Critic with Heuristic-Based Attention for Partially Observable Multi-Agent Path Finding.

## Setup

**Dependencies**

Create the virtual environment and install the required packages.

```
conda create -n sacha python=3.10
pip install -r requirement.txt
conda activate sacha
```

**Benchmarks**

Generate the test set used for evaluation.
```
cd benchmark
python create_test.py
```

## Train

**SACHA**

  ``python train.py``

**SACHA(C)**

  ``python train.py --communication``

## Evaluate

  ``python evaluate.py --load_from_dir path/to/dir``
