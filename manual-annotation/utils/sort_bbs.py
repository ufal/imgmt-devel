#!/usr/bin/env python3
"""Sort bounding boxes in an existing webapp data directory.

The sort is performed in place on the shared native-format BB files and the
alignment files.  Box IDs are preserved, while alignments are ordered by the
position of their source (SVG A) box.

Usage:
    python sort_bbs.py <data_dir> [--user USER_ID]
"""

import argparse
import json
import sys
from pathlib import Path


def _position_key(box: dict) -> tuple[float, float]:
    """Sort boxes from top to bottom, then left to right."""
    return (float(box["y"]), float(box["x"]))


def sort_bb_file(bb_file: Path) -> None:
    """Sort one native-format BB file in place."""
    with bb_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    boxes = data.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError(f"BB file has no boxes list: {bb_file}")

    boxes.sort(key=_position_key)
    with bb_file.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def sort_alignment_file(alignment_file: Path, data_dir: Path) -> None:
    """Sort alignments by the order of their source boxes in SVG A."""
    with alignment_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    alignments = data.get("alignments")
    if not isinstance(alignments, list):
        raise ValueError(f"Alignment file has no alignments list: {alignment_file}")

    bb_file = data_dir / "bbs" / data["image_id"] / f"{data['src_lang']}.json"
    if not bb_file.exists():
        print(
            f"Warning: source BB file not found, leaving alignments unchanged: {bb_file}",
            file=sys.stderr,
        )
        return
    with bb_file.open("r", encoding="utf-8") as fh:
        boxes = json.load(fh).get("boxes")
    if not isinstance(boxes, list):
        raise ValueError(f"BB file has no boxes list: {bb_file}")

    box_order = {
        str(box["id"]): index
        for index, box in enumerate(boxes)
        if "id" in box
    }
    missing_box_order = len(box_order)
    alignments.sort(
        key=lambda alignment: box_order.get(
            str(alignment.get("src_box")), missing_box_order
        )
    )
    with alignment_file.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def find_alignment_files(
    data_dir: Path, user_id: str | None = None
) -> list[Path]:
    """Return alignment files, optionally limited to pairs assigned to a user."""
    if user_id is None:
        return sorted((data_dir / "pairs").glob("*/alignments.json"))

    users_file = data_dir / "users.json"
    with users_file.open("r", encoding="utf-8") as fh:
        users = json.load(fh)
    if user_id not in users:
        raise ValueError(f"User not found in {users_file}: {user_id}")

    alignment_files = [
        data_dir / "pairs" / pair_id / "alignments.json"
        for pair_id in users[user_id].get("datasets", [])
    ]
    for alignment_file in alignment_files:
        if not alignment_file.exists():
            raise FileNotFoundError(f"Alignment file not found: {alignment_file}")
    return sorted(alignment_files)


def find_bb_files(data_dir: Path, user_id: str | None = None) -> list[Path]:
    """Return BB files, optionally limited to pairs assigned to a user."""
    if user_id is None:
        return sorted((data_dir / "bbs").glob("*/*.json"))

    bb_files: set[Path] = set()
    for alignment_file in find_alignment_files(data_dir, user_id):
        with alignment_file.open("r", encoding="utf-8") as fh:
            pair = json.load(fh)
        image_id = pair["image_id"]
        for language_key in ("src_lang", "tgt_lang"):
            bb_file = data_dir / "bbs" / image_id / f"{pair[language_key]}.json"
            if bb_file.exists():
                bb_files.add(bb_file)

    return sorted(bb_files)


def sort_bbs(data_dir: Path, user_id: str | None = None) -> int:
    """Sort BB and alignment files, optionally limited to one user's pairs."""
    bb_files = find_bb_files(data_dir, user_id)
    for bb_file in bb_files:
        if not bb_file.exists():
            raise FileNotFoundError(f"BB file not found: {bb_file}")
        sort_bb_file(bb_file)
    for alignment_file in find_alignment_files(data_dir, user_id):
        sort_alignment_file(alignment_file, data_dir)
    return len(bb_files)


def sort_bbs_for_user(data_dir: Path, user_id: str) -> int:
    """Sort BB files belonging to pairs assigned to *user_id*."""
    return sort_bbs(data_dir, user_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort bounding boxes in an existing webapp data directory."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Root of the webapp data directory (contains bbs/).",
    )
    parser.add_argument(
        "--user",
        dest="user_id",
        help="Sort only BBs belonging to pairs assigned to this user.",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"data directory not found: {args.data_dir}")

    count = sort_bbs(args.data_dir, args.user_id)
    scope = f" assigned to {args.user_id!r}" if args.user_id else ""
    print(f"Sorted {count} BB file(s){scope} in {args.data_dir / 'bbs'}")


if __name__ == "__main__":
    main()
