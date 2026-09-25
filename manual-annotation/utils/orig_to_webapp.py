#!/usr/bin/env python3
"""
orig_to_webapp.py — Convert one original-format record to the native webapp format.

The webapp reads bounding boxes and SVGs from a shared per-image store
(bbs/<image_id>/<lang>.*) and alignments from a per-pair file
(pairs/<pair_id>/alignments.json).  This script populates those files from an
original-format JSON record.

  <output_dir>/
      bbs/<image_id>/<lang>.json   — boxes + image metadata (created once per lang)
      bbs/<image_id>/<lang>.svg    — SVG (copied once per lang)
      pairs/<pair_id>/
          alignments.json          — 1:1 alignment indices + pair metadata

The same image may appear in several language pairs; storing BBs once avoids
duplication and ensures a single source of truth for BB annotation.  The bbs/
files are skipped (not overwritten) if they already exist.  Use sort_bbs.py
separately when boxes need to be ordered in an existing batch.

Original format (one JSON file per language-pair per image):
  <data_root>/<image_id>/<src>-<tgt>.json   — texts + bounding boxes
  <data_root>/<image_id>/svg/<src>.svg       — source SVG
  <data_root>/<image_id>/svg/<tgt>.svg       — target SVG

Usage:
    python orig_to_webapp.py <orig_json> <output_dir> [--pair-id ID]

Example:
    python orig_to_webapp.py ../orig_data/train/543/es-it.json ./data --pair-id pair_001
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def _write_bb_file(
    bb_file: Path,
    image_id: str,
    language: str,
    png_meta: dict,
    texts: list,
    bbs: list,
) -> None:
    """Write a BB file for one image/language; skips if the file already exists."""
    if bb_file.exists():
        return
    if len(texts) != len(bbs):
        raise ValueError(
            f"texts length ({len(texts)}) != bounding_boxes length ({len(bbs)}) "
            f"for image {image_id!r}, language {language!r}"
        )
    boxes = [
        {
            "id": str(i + 1),
            "x": bb["x"],
            "y": bb["y"],
            "w": bb["w"],
            "h": bb["h"],
            "text": text,
        }
        for i, (bb, text) in enumerate(zip(bbs, texts))
    ]
    data = {
        "image_id": image_id,
        "language": language,
        "png": png_meta,
        "boxes": boxes,
    }
    with bb_file.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def convert(
    orig_json: Path,
    output_dir: Path,
    pair_id: str,
    data_dir: Path | None = None,
) -> dict:
    """
    Convert one original-format JSON file to the native webapp layout.

    Writes:
      <output_dir>/bbs/<image_id>/<src_lang>.json   (skipped if exists)
      <output_dir>/bbs/<image_id>/<src_lang>.svg    (skipped if exists)
      <output_dir>/bbs/<image_id>/<tgt_lang>.json   (skipped if exists)
      <output_dir>/bbs/<image_id>/<tgt_lang>.svg    (skipped if exists)
      <output_dir>/pairs/<pair_id>/alignments.json

    Returns a dict suitable for an entry in mapping.json:
      {image_id, src_lang, tgt_lang, orig_json (relative to data_dir if given)}
    """
    with orig_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    image_id = orig_json.parent.name
    src_lang = data["source_language"]
    tgt_lang = data["target_language"]

    bbs_dir = output_dir / "bbs" / image_id
    bbs_dir.mkdir(parents=True, exist_ok=True)

    # Write BB files (one per language; idempotent — skips if already written)
    _write_bb_file(
        bbs_dir / f"{src_lang}.json",
        image_id,
        src_lang,
        data.get("source_PNG", {}),
        data["source_texts"],
        data["source_text_bounding_boxes"],
    )
    _write_bb_file(
        bbs_dir / f"{tgt_lang}.json",
        image_id,
        tgt_lang,
        data.get("target_PNG", {}),
        data["target_texts"],
        data["target_text_bounding_boxes"],
    )

    # Copy SVG files (once per language; skipped if already copied)
    svg_dir = orig_json.parent / "svg"
    for lang in (src_lang, tgt_lang):
        src_svg = svg_dir / f"{lang}.svg"
        dest_svg = bbs_dir / f"{lang}.svg"
        if src_svg.exists() and not dest_svg.exists():
            shutil.copy2(src_svg, dest_svg)
        elif not src_svg.exists():
            print(f"Warning: SVG not found: {src_svg}", file=sys.stderr)

    # Write alignment file (1:1 by index position)
    n_src = len(data["source_texts"])
    n_tgt = len(data["target_texts"])
    n = min(n_src, n_tgt)
    if n_src != n_tgt:
        print(
            f"Warning: {n_src} source and {n_tgt} target boxes; "
            f"only {n} alignment(s) produced.",
            file=sys.stderr,
        )

    pair_dir = output_dir / "pairs" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    alignment_data = {
        "image_id": image_id,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "alignment_method": data.get("alignment_method", ""),
        "alignments": [{"src_box": str(i + 1), "tgt_box": str(i + 1)} for i in range(n)],
    }
    with (pair_dir / "alignments.json").open("w", encoding="utf-8") as fh:
        json.dump(alignment_data, fh, ensure_ascii=False, indent=2)

    orig_json_str = (
        str(orig_json.relative_to(data_dir)) if data_dir else str(orig_json)
    )
    return {
        "image_id": image_id,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "orig_json": orig_json_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an original-format JSON record to the native webapp layout "
            "(bbs/<image_id>/<lang>.json + pairs/<pair_id>/alignments.json)."
        )
    )
    parser.add_argument("orig_json", type=Path, help="Path to the original JSON file.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root output directory (webapp data/ dir or batch root).",
    )
    parser.add_argument(
        "--pair-id",
        default="pair_001",
        help="Pair identifier to use (default: pair_001).",
    )
    args = parser.parse_args()

    if not args.orig_json.exists():
        print(f"Error: file not found: {args.orig_json}", file=sys.stderr)
        sys.exit(1)

    meta = convert(args.orig_json, args.output_dir, args.pair_id)
    print(
        f"Written to {args.output_dir} "
        f"(image {meta['image_id']!r}, {meta['src_lang']}-{meta['tgt_lang']}, "
        f"pair {args.pair_id!r})"
    )


if __name__ == "__main__":
    main()
