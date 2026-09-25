
## Setup

Install the dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

Set `IMGMT_DATA_DIR` to the directory containing the original ImgMT data. The
Makefile expects the original data and the annotations to have matching
`test/` and `dev/` subdirectories.

## Manual evaluation

Evaluate the completed test annotations with:

```bash
IMGMT_DATA_DIR=/path/to/orig_data make manual-eval
```

The evaluation reads the annotation JSON files and does not require SVG files.
It is therefore intentionally independent of SVG distribution.

### Distribute SVGs

SVGs can optionally be copied from the original ImgMT data into the shared
annotation directories:

```bash
IMGMT_DATA_DIR=/path/to/orig_data make distribute-svgs
```

The target above distributes SVGs for the test set. Use
`make distribute-svgs-dev` or `make distribute-svgs-test` to select a set.
Use `make check-svgs` (or `make check-svgs-dev` /
`make check-svgs-test`) to verify that every required SVG exists, is
byte-for-byte identical to the original, and that there are no excess SVGs,
without changing any files:

```bash
IMGMT_DATA_DIR=/path/to/orig_data make check-svgs
```
