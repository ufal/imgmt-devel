#!/usr/bin/env python3
"""
webapp_to_orig.py — Convert an annotated webapp batch pair back to the original format.

Reads from the native webapp data layout produced by orig_to_webapp.py / prepare_batch.py:

  <data_dir>/
      bbs/<image_id>/<lang>.json        — image metadata + bounding boxes (post-annotation)
      pairs/<pair_id>/alignments.json   — pair metadata + alignments (post-annotation)

Because the webapp writes BB corrections directly to the shared bbs/ files, the data
found there always reflects the latest state from any annotator.

The webapp allows n:m alignments; the original format only supports implicit 1:1
alignment by index.  Therefore:
  • Only boxes participating in strictly 1:1 alignments are emitted.
  • Multi-aligned or unaligned boxes are skipped with a warning.

Usage:
    python webapp_to_orig.py <data_dir> <pair_id> <output_json>

Arguments:
    data_dir     Root of the webapp data directory (contains bbs/ and pairs/).
    pair_id      Pair identifier (e.g. pair_001).
    output_json  Destination original-format JSON file.

Example:
    python webapp_to_orig.py ./data pair_001 ./out/543/es-it.json
"""

import argparse
import json
import sys
from pathlib import Path


def _collect_one_to_one(
    alignments: list[dict],
    a_key: str,
    b_key: str,
) -> tuple[dict[str, str], set[str], set[str]]:
    """
    Return:
      pairs   — {a_id: b_id} for strictly 1:1 aligned pairs
      skip_a  — a-IDs excluded (multi-aligned in the a-direction)
      skip_b  — b-IDs excluded (multi-aligned in the b-direction)
    """
    a_counts: dict[str, int] = {}
    b_counts: dict[str, int] = {}
    for aln in alignments:
        a_counts[aln[a_key]] = a_counts.get(aln[a_key], 0) + 1
        b_counts[aln[b_key]] = b_counts.get(aln[b_key], 0) + 1

    pairs: dict[str, str] = {}
    for aln in alignments:
        aid, bid = aln[a_key], aln[b_key]
        if a_counts[aid] == 1 and b_counts[bid] == 1:
            pairs[aid] = bid

    all_a = {aln[a_key] for aln in alignments}
    all_b = {aln[b_key] for aln in alignments}
    return pairs, all_a - set(pairs.keys()), all_b - set(pairs.values())


def _int_suffix(box_id: str) -> int:
    """Sort key: extract trailing integer from a box ID such as '3' or '12'."""
    return int(box_id) if box_id.isdigit() else 0


def convert(
    data_dir: Path,
    pair_id: str,
    output_json: Path,
) -> None:
    """
    Convert one batch pair back to the original JSON format.

    Box data and alignments are read from the shared bbs/ and pairs/ directories,
    which always contain the latest post-annotation state written by the webapp.
    """
    # --- Load pair metadata ---------------------------------------------------
    aln_file = data_dir / "pairs" / pair_id / "alignments.json"
    if not aln_file.exists():
        raise FileNotFoundError(f"Alignment file not found: {aln_file}")
    with aln_file.open("r", encoding="utf-8") as fh:
        aln_data = json.load(fh)

    image_id = aln_data["image_id"]
    src_lang = aln_data["src_lang"]
    tgt_lang = aln_data["tgt_lang"]
    alignment_method = aln_data.get("alignment_method", "manual")

    bbs_dir = data_dir / "bbs" / image_id

    # --- Load box data from shared bbs/ files ---------------------------------
    for lang in (src_lang, tgt_lang):
        bb_path = bbs_dir / f"{lang}.json"
        if not bb_path.exists():
            raise FileNotFoundError(f"BB file not found: {bb_path}")

    with (bbs_dir / f"{src_lang}.json").open("r", encoding="utf-8") as fh:
        src_bb_file = json.load(fh)
    with (bbs_dir / f"{tgt_lang}.json").open("r", encoding="utf-8") as fh:
        tgt_bb_file = json.load(fh)

    src_png = src_bb_file.get("png", {})
    tgt_png = tgt_bb_file.get("png", {})

    # --- Resolve 1:1 alignments from pairs/ ----------------------------------
    src_boxes_raw = {b["id"]: b for b in src_bb_file["boxes"]}
    tgt_boxes_raw = {b["id"]: b for b in tgt_bb_file["boxes"]}
    alignments_raw = aln_data["alignments"]

    pairs, skip_a, skip_b = _collect_one_to_one(alignments_raw, "src_box", "tgt_box")

    for aid in sorted(skip_a):
        print(f"Warning: src-box {aid!r} is multi-aligned — excluded.", file=sys.stderr)
    for bid in sorted(skip_b):
        print(f"Warning: tgt-box {bid!r} is multi-aligned — excluded.", file=sys.stderr)

    all_src_ids = {b["id"] for b in src_bb_file["boxes"]}
    all_tgt_ids = {b["id"] for b in tgt_bb_file["boxes"]}
    aligned_src = {aln["src_box"] for aln in alignments_raw}
    aligned_tgt = {aln["tgt_box"] for aln in alignments_raw}
    for aid in sorted(all_src_ids - aligned_src):
        print(f"Warning: src-box {aid!r} has no alignment — excluded.", file=sys.stderr)
    for bid in sorted(all_tgt_ids - aligned_tgt):
        print(f"Warning: tgt-box {bid!r} has no alignment — excluded.", file=sys.stderr)

    ordered_src_ids = sorted(pairs.keys(), key=_int_suffix)
    src_texts, src_bbs_out, tgt_texts, tgt_bbs_out = [], [], [], []
    for sid in ordered_src_ids:
        tid = pairs[sid]
        bs = src_boxes_raw[sid]
        bt = tgt_boxes_raw[tid]
        src_texts.append(bs["text"])
        src_bbs_out.append({"x": bs["x"], "y": bs["y"], "w": bs["w"], "h": bs["h"]})
        tgt_texts.append(bt["text"])
        tgt_bbs_out.append({"x": bt["x"], "y": bt["y"], "w": bt["w"], "h": bt["h"]})

    # --- Write output ---------------------------------------------------------
    output = {
        "alignment_method": alignment_method,
        "source_language": src_lang,
        "source_PNG": src_png,
        "source_texts": src_texts,
        "source_text_bounding_boxes": src_bbs_out,
        "target_language": tgt_lang,
        "target_PNG": tgt_png,
        "target_texts": tgt_texts,
        "target_text_bounding_boxes": tgt_bbs_out,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=4)

    print(f"Written {len(src_texts)} aligned pair(s) to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an annotated batch pair back to the original JSON format."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Root of the webapp data directory (contains bbs/ and pairs/).",
    )
    parser.add_argument("pair_id", help="Pair identifier (e.g. pair_001).")
    parser.add_argument("output_json", type=Path, help="Destination original-format JSON file.")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    convert(args.data_dir, args.pair_id, args.output_json)


if __name__ == "__main__":
    main()

