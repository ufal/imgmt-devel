import argparse
import glob
import logging
import os
from pathlib import Path
from shapely.ops import unary_union

from eval.align_bb import bounding_box_alignment 
from eval.alignment_readers import read_orig, read_webapp, get_orig_pair_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgmt-dir", help="Path to original data")
    parser.add_argument("--manual-dir", help="Path to manual manually annotated subset")
    parser.add_argument("--skip-ids", nargs="*", default=[
        "pair_004", "pair_013", "pair_019", "pair_037", # different by MN
        "pair_012", "pair_015", "pair_020", "pair_031", "pair_034", "pair_044", # not assessed by MN
        "pair_055", "pair_064", "pair_068", "pair_073", "pair_080", "pair_083", "pair_098", "pair_099", # different by SU
        "pair_057", "pair_061", "pair_076", "pair_092", # not assessed by SU
      ], help="List of example IDs to skip")
    parser.add_argument("--average", choices=["macro", "micro"], default="macro", help="Averaging method for stats calculation")
    return parser.parse_args()

def _init_counts():
    return {
        "total_image_pairs": 0,
        "total_images": 0,
        "auto_bb_count": 0,
        "manual_bb_count": 0,
        "auto_matched_bb_count": 0,
        "manual_matched_bb_count": 0,
        "auto_bb_area": 0.0,
        "manual_bb_area": 0.0,
        "auto_matched_bb_area": 0.0,
        "manual_matched_bb_area": 0.0,
        "auto_xling_matched_count": 0,
        "manual_xling_matched_count": 0,
        "auto_xling_matched_area": 0.0,
        "manual_xling_matched_area": 0.0,
        "auto_xling_match_correct_count": 0,
        "manual_xling_match_correct_count": 0,
        "auto_xling_match_correct_area": 0.0,
        "manual_xling_match_correct_area": 0.0,
    }

def _convert_bbs(bbs):
    '''
    Convert bounding box coordinates to the [x1, y1, x2, y2] format.
    '''
    for bb in bbs:
        yield [bb['x'], bb['y'], bb['x'] + bb['width'], bb['y'] + bb['height']]


def _prepare_bbs_lists(auto_data, manual_data):
    auto_bbs = {side: [bb['bbox'] for bb in auto_data[f"svg{side}"]["boxes"]] for side in ["A", "B"]}
    manual_bbs = {side: [bb['bbox'] for bb in manual_data[f"svg{side}"]["boxes"]] for side in ["A", "B"]}
    return auto_bbs, manual_bbs

def _prepare_auto_manual_indexes(auto_bbs, manual_bbs):
    auto_idx2manual_idxs = {}
    manual_idx2auto_idxs = {}
    for side in ["A", "B"]:
        auto_manual_bb_matching = bounding_box_alignment([bb.bounds for bb in auto_bbs[side]], [bb.bounds for bb in manual_bbs[side]])
        auto_idx2manual_idxs[side] = {}
        for auto_idx, manual_idx in auto_manual_bb_matching:
            auto_idx2manual_idxs[side].setdefault(auto_idx, []).append(manual_idx)
        manual_idx2auto_idxs[side] = {}
        for auto_idx, manual_idx in auto_manual_bb_matching:
            manual_idx2auto_idxs[side].setdefault(manual_idx, []).append(auto_idx)
    return auto_idx2manual_idxs, manual_idx2auto_idxs

def _prepare_xling_indexes(auto_data, manual_data):
    auto_bbid2idx = {}
    manual_bbid2idx = {}
    for side in ["A", "B"]:
        auto_bbid2idx[side] = {bb['id']: idx for idx, bb in enumerate(auto_data[f"svg{side}"]["boxes"])}
        manual_bbid2idx[side] = {bb['id']: idx for idx, bb in enumerate(manual_data[f"svg{side}"]["boxes"])}
    
    auto_xling_aligns = auto_data.get('alignments', [])
    manual_xling_aligns = manual_data.get('alignments', [])

    auto_xling_idx_dict = {}
    manual_xling_idx_dict = {}
    for direction in [('A', 'B'), ('B', 'A')]:
        side_from, side_to = direction

        auto_xling_idx_dict[side_from] = {}
        for align in auto_xling_aligns:
            auto_from_idx = auto_bbid2idx[side_from].get(align[f"box{side_from}"])
            auto_to_idx = auto_bbid2idx[side_to].get(align[f"box{side_to}"])
            if auto_from_idx is not None and auto_to_idx is not None:
                # cross-lingual alignments are guaranteed to be one-to-one, so we can store them in a dictionary
                auto_xling_idx_dict[side_from][auto_from_idx] = auto_to_idx

        manual_xling_idx_dict[side_from] = {}
        for align in manual_xling_aligns:
            manual_from_idx = manual_bbid2idx[side_from].get(align[f"box{side_from}"])
            manual_to_idx = manual_bbid2idx[side_to].get(align[f"box{side_to}"])
            if manual_from_idx is not None and manual_to_idx is not None:
                # cross-lingual alignments are guaranteed to be one-to-one, so we can store them in a dictionary
                manual_xling_idx_dict[side_from][manual_from_idx] = manual_to_idx

    return auto_xling_idx_dict, manual_xling_idx_dict

def _calculate_overlap_area(src_bbs, tgt_bbs, side, src_idx, tgt_idxs):
    return unary_union([tgt_bbs[side][idx] for idx in tgt_idxs]).intersection(src_bbs[side][src_idx]).area

def _are_bbs_same_across_xling_alignment(src_a_idx, tgt_a_idxs, src_a2b_idx, tgt_a2b_idx, src2tgt_b_idx):
    if src_a_idx is None or not tgt_a_idxs:
        return False
    src_b_idx = src_a2b_idx.get(src_a_idx)
    if src_b_idx is None:
        return False
    tgt_b_idxs = [tgt_a2b_idx[tgt_a_idx] for tgt_a_idx in tgt_a_idxs if tgt_a_idx in tgt_a2b_idx]
    if not tgt_b_idxs:
        return False
    tgt_b_idxs_from_src_b = src2tgt_b_idx.get(src_b_idx)
    if tgt_b_idxs_from_src_b is None:
        return False
    return set(tgt_b_idxs) == set(tgt_b_idxs_from_src_b)

def _update_counts(counts, auto_data, manual_data):
    # create auxiliary indexes for bounding boxes
    auto_bbs, manual_bbs = _prepare_bbs_lists(auto_data, manual_data)
    auto_idx2manual_idxs, manual_idx2auto_idxs = _prepare_auto_manual_indexes(auto_bbs, manual_bbs)
    auto_xling_idx_dict, manual_xling_idx_dict = _prepare_xling_indexes(auto_data, manual_data)

    counts["total_image_pairs"] += 1
    for direction in [('A', 'B'), ('B', 'A')]:
        side_from, side_to = direction
        counts["total_images"] += 1  # Assuming each pair has two images

        counts["auto_bb_count"] += len(auto_bbs[side_from])
        counts["manual_bb_count"] += len(manual_bbs[side_from])
        counts["auto_bb_area"] += sum(bb.area for bb in auto_bbs[side_from])
        counts["manual_bb_area"] += sum(bb.area for bb in manual_bbs[side_from])

        # count statistics for bounding box matching
        counts["auto_matched_bb_count"] += len(auto_idx2manual_idxs[side_from].keys())
        counts["manual_matched_bb_count"] += len(manual_idx2auto_idxs[side_from].keys())
        counts["auto_matched_bb_area"] += sum(_calculate_overlap_area(auto_bbs, manual_bbs, side_from, auto_idx, manual_idxs) for auto_idx, manual_idxs in auto_idx2manual_idxs[side_from].items())
        counts["manual_matched_bb_area"] += sum(_calculate_overlap_area(manual_bbs, auto_bbs, side_from, manual_idx, auto_idxs) for manual_idx, auto_idxs in manual_idx2auto_idxs[side_from].items())

        counts["auto_xling_matched_count"] += sum(1 for auto_idx in auto_xling_idx_dict[side_from] if auto_idx in auto_idx2manual_idxs[side_from])
        counts["manual_xling_matched_count"] += sum(1 for manual_idx in manual_xling_idx_dict[side_from] if manual_idx in manual_idx2auto_idxs[side_from])
        counts["auto_xling_matched_area"] += sum(_calculate_overlap_area(auto_bbs, manual_bbs, side_from, auto_idx, manual_idxs) for auto_idx, manual_idxs in auto_idx2manual_idxs[side_from].items() if auto_idx in auto_xling_idx_dict[side_from])
        counts["manual_xling_matched_area"] += sum(_calculate_overlap_area(manual_bbs, auto_bbs, side_from, manual_idx, auto_idxs) for manual_idx, auto_idxs in manual_idx2auto_idxs[side_from].items() if manual_idx in manual_xling_idx_dict[side_from])

        counts["auto_xling_match_correct_count"] += sum(1 for auto_idx, manual_idxs in auto_idx2manual_idxs[side_from].items() if _are_bbs_same_across_xling_alignment(auto_idx, manual_idxs, auto_xling_idx_dict[side_from], manual_xling_idx_dict[side_from], auto_idx2manual_idxs[side_to]))
        counts["manual_xling_match_correct_count"] += sum(1 for manual_idx, auto_idxs in manual_idx2auto_idxs[side_from].items() if _are_bbs_same_across_xling_alignment(manual_idx, auto_idxs, manual_xling_idx_dict[side_from], auto_xling_idx_dict[side_from], manual_idx2auto_idxs[side_to]))
        counts["auto_xling_match_correct_area"] += sum(_calculate_overlap_area(auto_bbs, manual_bbs, side_from, auto_idx, manual_idxs) for auto_idx, manual_idxs in auto_idx2manual_idxs[side_from].items() if _are_bbs_same_across_xling_alignment(auto_idx, manual_idxs, auto_xling_idx_dict[side_from], manual_xling_idx_dict[side_from], auto_idx2manual_idxs[side_to]))
        counts["manual_xling_match_correct_area"] += sum(_calculate_overlap_area(manual_bbs, auto_bbs, side_from, manual_idx, auto_idxs) for manual_idx, auto_idxs in manual_idx2auto_idxs[side_from].items() if _are_bbs_same_across_xling_alignment(manual_idx, auto_idxs, manual_xling_idx_dict[side_from], auto_xling_idx_dict[side_from], manual_idx2auto_idxs[side_to]))

    # count statistics for cross-lingual bounding box alignment
    #auto_a_bbid2bb = {bb['id']: bb['bbox'] for bb in auto_data["svgA"]["boxes"]}
    #auto_b_bbid2bb = {bb['id']: bb['bbox'] for bb in auto_data["svgB"]["boxes"]}
    #auto_xling_aligns = for align in auto_data['alignments']

        

def _print_counts(counts):
    logging.info(f"Total image pairs processed: {counts['total_image_pairs']}")
    logging.info(f"Total images processed: {counts['total_images']}")
    logging.info(f"Total auto bounding boxes: {counts['auto_bb_count']}")
    logging.info(f"Total manual bounding boxes: {counts['manual_bb_count']}")
    logging.info(f"Total auto bounding boxes matched in manual data: {counts['auto_matched_bb_count']}")
    logging.info(f"Total manual bounding boxes matched in auto data: {counts['manual_matched_bb_count']}")
    logging.info(f"Total area of auto bounding boxes: {counts['auto_bb_area']}")
    logging.info(f"Total area of manual bounding boxes: {counts['manual_bb_area']}")
    logging.info(f"Total area of auto bounding boxes matched in manual data: {counts['auto_matched_bb_area']}")
    logging.info(f"Total area of manual bounding boxes matched in auto data: {counts['manual_matched_bb_area']}")
    logging.info(f"Total auto bounding boxes cross-lingually matched: {counts['auto_xling_matched_count']}")
    logging.info(f"Total manual bounding boxes cross-lingually matched: {counts['manual_xling_matched_count']}")
    logging.info(f"Total area of auto bounding boxes cross-lingually matched: {counts['auto_xling_matched_area']}")
    logging.info(f"Total area of manual bounding boxes cross-lingually matched: {counts['manual_xling_matched_area']}")
    logging.info(f"Total auto bounding boxes cross-lingually matched correctly: {counts['auto_xling_match_correct_count']}")
    logging.info(f"Total manual bounding boxes cross-lingually matched correctly: {counts['manual_xling_match_correct_count']}")
    logging.info(f"Total area of auto bounding boxes cross-lingually matched correctly: {counts['auto_xling_match_correct_area']}")
    logging.info(f"Total area of manual bounding boxes cross-lingually matched correctly: {counts['manual_xling_match_correct_area']}")

def calculate_stats(counts):
    """
    Calculate and return statistics based on the counts dictionary.
    """
    def _fscore(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    stats = {}

    stats['bb_count_precision'] = counts['auto_matched_bb_count'] / counts['auto_bb_count'] if counts['auto_bb_count'] > 0 else 0
    stats['bb_count_recall'] = counts['manual_matched_bb_count'] / counts['manual_bb_count'] if counts['manual_bb_count'] > 0 else 0
    stats['bb_count_fscore'] = _fscore(stats['bb_count_precision'], stats['bb_count_recall'])

    stats['bb_area_precision'] = counts['auto_matched_bb_area'] / counts['auto_bb_area'] if counts['auto_bb_area'] > 0 else 0
    stats['bb_area_recall'] = counts['manual_matched_bb_area'] / counts['manual_bb_area'] if counts['manual_bb_area'] > 0 else 0
    stats['bb_area_fscore'] = _fscore(stats['bb_area_precision'], stats['bb_area_recall'])

    stats['xling_count_precision'] = counts['auto_xling_matched_count'] / counts['auto_bb_count'] if counts['auto_bb_count'] > 0 else 0
    stats['xling_count_recall'] = counts['manual_xling_matched_count'] / counts['manual_bb_count'] if counts['manual_bb_count'] > 0 else 0
    stats['xling_count_fscore'] = _fscore(stats['xling_count_precision'], stats['xling_count_recall'])

    stats['xling_area_precision'] = counts['auto_xling_matched_area'] / counts['auto_bb_area'] if counts['auto_bb_area'] > 0 else 0
    stats['xling_area_recall'] = counts['manual_xling_matched_area'] / counts['manual_bb_area'] if counts['manual_bb_area'] > 0 else 0
    stats['xling_area_fscore'] = _fscore(stats['xling_area_precision'], stats['xling_area_recall'])

    stats['xling_correct_count_precision'] = counts['auto_xling_match_correct_count'] / counts['auto_bb_count'] if counts['auto_bb_count'] > 0 else 0
    stats['xling_correct_count_recall'] = counts['manual_xling_match_correct_count'] / counts['manual_bb_count'] if counts['manual_bb_count'] > 0 else 0
    stats['xling_correct_count_fscore'] = _fscore(stats['xling_correct_count_precision'], stats['xling_correct_count_recall'])

    stats['xling_correct_area_precision'] = counts['auto_xling_match_correct_area'] / counts['auto_bb_area'] if counts['auto_bb_area'] > 0 else 0
    stats['xling_correct_area_recall'] = counts['manual_xling_match_correct_area'] / counts['manual_bb_area'] if counts['manual_bb_area'] > 0 else 0
    stats['xling_correct_area_fscore'] = _fscore(stats['xling_correct_area_precision'], stats['xling_correct_area_recall'])

    return stats

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()
    orig_path = Path(args.imgmt_dir)
    manual_path = Path(args.manual_dir)

    counts = _init_counts()

    stats_per_example = []

    # iterate over all directories in the manual_dir/pairs directory
    for manual_ex_path in sorted(glob.glob(os.path.join(manual_path, "pairs", "*"))):
        manual_ex_id = Path(manual_ex_path).name
        if manual_ex_id in args.skip_ids:
            logging.info(f"Skipping manual example: {manual_ex_id}")
            continue
        logging.info(f"Processing manual example: {manual_ex_id}")
        manual_ex_data = read_webapp(manual_path, manual_ex_id, normalize_bb=True)
        #print(f"Manual example data: {manual_ex_data}")
        orig_ex_id = get_orig_pair_id(manual_ex_data)
        orig_ex_data = read_orig(orig_path, orig_ex_id, normalize_bb=True)
        _update_counts(counts, orig_ex_data, manual_ex_data)
        _print_counts(counts)
        if args.average == "macro":
            stats_per_example.append(calculate_stats(counts))
            counts = _init_counts()  # Reset counts for macro averaging

    stats = {}
    if args.average == "micro":
        stats = calculate_stats(counts)
    elif args.average == "macro":
        # Calculate average stats across all examples
        for metric in stats_per_example[0].keys():
            stats[metric] = sum(example[metric] for example in stats_per_example) / len(stats_per_example)
    for metric, value in stats.items():
        logging.info(f"{metric}: {value}")

if __name__ == "__main__":
    main()
