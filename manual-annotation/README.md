
### Distribute SVGs

Copy missing SVGs from the original ImgMT data into the shared annotation
directories:

```bash
python utils/distribute_svgs.py /path/to/orig_data /path/to/my-data
```

Use `--check` to verify that every required SVG exists, is byte-for-byte
identical to the original, and that there are no excess SVGs, without changing
any files:

```bash
python utils/distribute_svgs.py /path/to/orig_data /path/to/my-data --check
```
