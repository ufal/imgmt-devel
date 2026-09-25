#!/usr/bin/env python3
"""
prepare_batch.py — Distribute image pairs across annotators and prepare the batch.

The script:
  1. Discovers all original-format JSON files under <data_dir>.
  2. Randomly samples language-pair JSONs across all image groups.
  3. Assigns each image-language SVG to at most one annotator, until each
     annotator has up to <n_pairs_per_annotator> pairs.
   4. Converts each assigned pair to the native webapp layout (bbs/, pairs/).
  5. Writes users.json (webapp user config) and mapping.json (pair → original
     file mapping) at the output root.

The output directory can be used directly as the webapp's data/ directory:

  <output_dir>/
      bbs/
          <image_id>/<lang>.json   — boxes + image metadata (shared across pairs)
          <image_id>/<lang>.svg    — SVG (shared)
      pairs/
          <pair_id>/
              alignments.json      — alignment indices + pair metadata
      users.json                   — webapp user/dataset config
      mapping.json                 — pair_id → {image_id, langs, annotator, orig_json}

Because bbs/ files are shared, any BB correction made through the webapp is
visible in every pair that references the same image, regardless of which
annotator is viewing it.

Usage:
    python prepare_batch.py <data_dir> <n_pairs_per_annotator> <output_dir>
                            [--annotators ann1 ann2 ...]
                            [--seed SEED]

Arguments:
    data_dir               Root of original data (e.g. orig_data/train).
    n_pairs_per_annotator  Maximum pairs per annotator.
    output_dir             Destination directory (becomes the webapp's data/).

Options:
    --annotators  Space-separated list of annotator IDs (default: annotator1).
    --seed        Random seed for reproducible sampling.

Example:
    python prepare_batch.py ../orig_data/train 10 ./data \\
        --annotators alice bob carol --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).parent
sys.path.insert(0, str(_UTILS_DIR))
from orig_to_webapp import convert  # noqa: E402


def find_orig_jsons(data_dir: Path) -> list[Path]:
    """Return all original-format JSON files found under data_dir."""
    return sorted(
        p for p in data_dir.rglob("*.json") if (p.parent / "svg").is_dir()
    )


def _assign_pairs(
    all_jsons: list[Path],
    annotators: list[str],
    n_per_annotator: int,
) -> dict[str, list[Path]]:
    """
    Assign language-pair JSON files to annotators ensuring no image-language SVG
    is shared across annotators. JSON files are shuffled before assignment so
    records from all image groups can be sampled. Multiple annotators may use
    one image group when their pairs use disjoint language SVGs.

    Returns {annotator: [list of assigned json_files]}.
    """
    assignment: dict[str, list[Path]] = {ann: [] for ann in annotators}
    svg_owners: dict[tuple[str, str], str] = {}
    shuffled_jsons = list(all_jsons)
    random.shuffle(shuffled_jsons)

    for json_file in shuffled_jsons:
        image_id = json_file.parent.name
        with json_file.open("r", encoding="utf-8") as fh:
            pair_data = json.load(fh)
        languages = {
            pair_data["source_language"],
            pair_data["target_language"],
        }

        # A pair is eligible for an annotator if every SVG it uses is either
        # unclaimed or already claimed by that same annotator.
        eligible = [
            ann
            for ann in annotators
            if len(assignment[ann]) < n_per_annotator
            and all(
                svg_owners.get((image_id, language), ann) == ann
                for language in languages
            )
        ]
        if not eligible:
            continue

        owner = min(eligible, key=lambda ann: len(assignment[ann]))
        assignment[owner].append(json_file)
        for language in languages:
            svg_owners.setdefault((image_id, language), owner)

    return assignment


def prepare_batch(
    data_dir: Path,
    n_per_annotator: int,
    output_dir: Path,
    annotators: list[str],
    seed: int | None = None,
) -> None:
    all_jsons = find_orig_jsons(data_dir)
    if not all_jsons:
        print(f"Error: no original-format JSON files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    if seed is not None:
        random.seed(seed)

    assignment = _assign_pairs(all_jsons, annotators, n_per_annotator)

    total_pairs = sum(len(v) for v in assignment.values())
    total_available = len(all_jsons)
    if total_pairs < len(annotators) * n_per_annotator:
        print(
            f"Warning: only {total_available} record(s) available across "
            f"{len(set(jf.parent.name for jf in all_jsons))} image(s); "
            f"some annotators may receive fewer than {n_per_annotator} pair(s).",
            file=sys.stderr,
        )

    mapping: dict[str, dict] = {}
    users: dict[str, dict] = {}

    # Global pair counter so pair IDs are unique across annotators
    pair_counter = 1

    for ann in annotators:
        pairs_for_ann = assignment[ann]
        pair_ids_for_ann: list[str] = []

        for json_file in pairs_for_ann:
            pair_id = f"pair_{pair_counter:03d}"
            pair_counter += 1

            print(
                f"  [{ann}] {json_file.relative_to(data_dir)}  →  {pair_id}"
            )
            meta = convert(json_file, output_dir, pair_id, data_dir=data_dir)
            meta["annotator"] = ann
            mapping[pair_id] = meta
            pair_ids_for_ann.append(pair_id)

        users[ann] = {"display_name": ann, "datasets": pair_ids_for_ann}

    # Write users.json
    users_file = output_dir / "users.json"
    with users_file.open("w", encoding="utf-8") as fh:
        json.dump(users, fh, ensure_ascii=False, indent=2)

    # Write mapping.json
    mapping_file = output_dir / "mapping.json"
    with mapping_file.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)

    print(f"\nDone.  Batch written to: {output_dir}")
    print(f"  users.json:   {users_file}")
    print(f"  mapping.json: {mapping_file}")
    for ann, pair_ids in [(a, [k for k, v in mapping.items() if v["annotator"] == a]) for a in annotators]:
        print(f"  {ann}: {len(pair_ids)} pair(s) — {', '.join(pair_ids) or '(none)'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribute image pairs across annotators and prepare a webapp batch."
    )
    parser.add_argument("data_dir", type=Path, help="Root directory of original data.")
    parser.add_argument(
        "n_pairs_per_annotator",
        type=int,
        help="Maximum number of image pairs per annotator.",
    )
    parser.add_argument("output_dir", type=Path, help="Destination directory (becomes the webapp's data/).")
    parser.add_argument(
        "--annotators",
        nargs="+",
        default=["annotator1"],
        metavar="ID",
        help="Annotator IDs (default: annotator1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    if args.n_pairs_per_annotator < 1:
        print("Error: n_pairs_per_annotator must be a positive integer.", file=sys.stderr)
        sys.exit(1)
    if len(args.annotators) != len(set(args.annotators)):
        print("Error: duplicate annotator IDs.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Preparing batch for {len(args.annotators)} annotator(s), "
        f"up to {args.n_pairs_per_annotator} pair(s) each …"
    )
    prepare_batch(
        args.data_dir,
        args.n_pairs_per_annotator,
        args.output_dir,
        args.annotators,
        args.seed,
    )


if __name__ == "__main__":
    main()
