# Installation

## Requirements

We recommend **Python 3.10+, [PyTorch 2.7.0+](https://pytorch.org/get-started/locally/), [transformers](https://github.com/huggingface/transformers) v4.51.3+**.

## Install via pip

```bash
pip install outformer
```

## Install from source
```bash
git clone https://github.com/milistu/outformer.git
cd outformer
pip install -e .
```

## Development Installation
If you want to contribute or modify the library:
```bash
git clone https://github.com/milistu/outformer.git
cd outformer
pip install -e ".[dev]"
```

## Verify Installation
```Python
import outformer
print(outformer.__version__)
```