#!/usr/bin/env python3
"""Copy missing ImgMT SVGs into the webapp's shared annotation directories.

Usage:
    python distribute_svgs.py <orig_data_dir> <annotation_data_dir>
    python distribute_svgs.py <orig_data_dir> <annotation_data_dir> --check

The original data is expected to contain directories such as
``<image_id>/svg/<language>.svg`` (possibly below one or more partition
directories).  The annotation data contains the corresponding
``bbs/<image_id>/`` directories.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path


def _source_svgs(orig_data_dir: Path) -> dict[tuple[str, str], Path]:
    """Return source SVGs indexed by image ID and language."""
    sources: dict[tuple[str, str], Path] = {}
    for svg_dir in orig_data_dir.rglob("svg"):
        if not svg_dir.is_dir():
            continue
        image_id = svg_dir.parent.name
        for source in svg_dir.glob("*.svg"):
            key = (image_id, source.stem)
            if key in sources and sources[key] != source:
                raise ValueError(
                    f"Multiple source SVGs found for {image_id}/{source.stem}: "
                    f"{sources[key]} and {source}"
                )
            sources[key] = source
    return sources


def distribute_svgs(
    orig_data_dir: Path, annotation_data_dir: Path, check: bool = False
) -> tuple[int, int]:
    """Copy missing SVGs and return ``(copied, problems)``.

    In checking mode no files are changed.  A problem is a missing destination
    SVG, a destination whose contents differ from the source, an excess
    destination SVG, or a missing source SVG for an annotation directory.
    """
    bbs_dir = annotation_data_dir / "bbs"
    sources = _source_svgs(orig_data_dir)
    copied = 0
    problems = 0

    for image_dir in sorted(path for path in bbs_dir.iterdir() if path.is_dir()):
        image_id = image_dir.name
        required_languages = {
            path.stem for path in image_dir.glob("*.json") if path.is_file()
        }
        if check:
            for destination in sorted(
                path for path in image_dir.glob("*.svg") if path.is_file()
            ):
                if destination.stem not in required_languages:
                    print(f"Excess annotation SVG: {destination}", file=sys.stderr)
                    problems += 1
        for language in sorted(required_languages):
            source = sources.get((image_id, language))
            destination = image_dir / f"{language}.svg"
            if source is None:
                print(
                    f"Missing source SVG for {image_id}/{language} in {orig_data_dir}",
                    file=sys.stderr,
                )
                problems += 1
            elif not destination.exists():
                if check:
                    print(f"Missing annotation SVG: {destination}", file=sys.stderr)
                    problems += 1
                else:
                    shutil.copy2(source, destination)
                    copied += 1
                    print(f"Copied {source} -> {destination}")
            elif not filecmp.cmp(source, destination, shallow=False):
                print(
                    f"Different SVG files: source {source}, annotation {destination}",
                    file=sys.stderr,
                )
                problems += 1

    return copied, problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribute missing original ImgMT SVGs to annotation bbs directories."
    )
    parser.add_argument("orig_data_dir", type=Path, help="Root of the original ImgMT data.")
    parser.add_argument(
        "annotation_data_dir",
        type=Path,
        help="Root of the annotation data (containing bbs/).",
    )
    parser.add_argument(
        "--check",
        "--checking",
        dest="check",
        action="store_true",
        help="Only check that required SVGs exist, match the originals, and have no excess SVGs.",
    )
    args = parser.parse_args()

    if not args.orig_data_dir.is_dir():
        parser.error(f"original data directory not found: {args.orig_data_dir}")
    if not (args.annotation_data_dir / "bbs").is_dir():
        parser.error(f"annotation bbs directory not found: {args.annotation_data_dir / 'bbs'}")

    try:
        copied, problems = distribute_svgs(
            args.orig_data_dir, args.annotation_data_dir, check=args.check
        )
    except ValueError as error:
        parser.error(str(error))

    if args.check:
        print(f"Checked annotation SVGs: {problems} problem(s).")
    else:
        print(f"Copied {copied} missing SVG(s); found {problems} problem(s).")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
