# Installation

```sh
pip install -r requirements.txt
```

# Usage

On a single image.

```sh
python poisson.py path/to/input path/to/output
```

Optional arguments:
- `-d` is the dilation parameter (positive integer), defaults to `-d 2`.
- `--equalize` to balance the output contrast, doesn't apply by default.