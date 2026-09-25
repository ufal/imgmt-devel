
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

## Prepare and sort annotations

The initial annotation batch can be generated from original data with:

```bash
IMGMT_DATA_DIR=/path/to/orig_data \
N_PAIRS_PER_ANNOTATOR=10 \
ANNOTATORS="alice bob carol" \
SEED=42 \
make prepare-batch
```

This writes the test batch to `annots/test`. Use `make prepare-batch-dev` to
prepare a batch from the dev set instead. The output directory is populated
with the webapp's `bbs/`, `pairs/`, `users.json`, and `mapping.json` files.
`ANNOTATORS` and `SEED` are optional; the defaults are `annotator1` and a
non-deterministic sample.

To sort the shared bounding-box files and their alignments after annotation:

```bash
make sort-bbs
```

Use `make sort-bbs-dev` or `make sort-bbs-test` to select a set, and
`python utils/sort_bbs.py annots/test --user alice` to sort only one
annotator's pairs.

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
