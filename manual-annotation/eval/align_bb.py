import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box
from shapely.ops import unary_union

from pathlib import Path

from eval.alignment_readers import read_orig, read_webapp, get_orig_pair_id

def set_iou(boxes_a, boxes_b):
    """Calculates the Intersection over Union (IoU) between two sets of bounding boxes."""
    if not boxes_a and not boxes_b:
        return 0.0
    
    poly_a = unary_union([box(*b) for b in boxes_a]) if boxes_a else box(0, 0, 0, 0).buffer(0)
    poly_b = unary_union([box(*b) for b in boxes_b]) if boxes_b else box(0, 0, 0, 0).buffer(0)
    
    if poly_a.is_empty and poly_b.is_empty:
        return 0.0
    
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    
    if union == 0:
        return 0.0
    return inter / union

def internal_iou(boxes):
    """Calculates the sum of pairwise internal IoUs for a set of bounding boxes."""
    if len(boxes) < 2:
        return 0.0
    total_iou = 0.0
    for b1, b2 in itertools.combinations(boxes, 2):
        total_iou += set_iou([b1], [b2])
    return total_iou

def partition_boxes(A, B):
    """
    Partitions the union of sets A and B into disjoint components 
    such that there is no bounding box overlap across partitions.
    """
    all_items = []
    for idx, b in enumerate(A):
        all_items.append({'box': b, 'index': idx, 'set': 'A'})
    for idx, b in enumerate(B):
        all_items.append({'box': b, 'index': idx, 'set': 'B'})
        
    n = len(all_items)
    adj = {i: [] for i in range(n)}
    
    for i in range(n):
        poly_i = box(*all_items[i]['box'])
        for j in range(i + 1, n):
            poly_j = box(*all_items[j]['box'])
            if poly_i.intersects(poly_j) and poly_i.intersection(poly_j).area > 0:
                adj[i].append(j)
                adj[j].append(i)
                    
    visited = [False] * n
    partitions = []
    
    for i in range(n):
        if not visited[i]:
            component = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            Ai_partition = [all_items[idx] for idx in component if all_items[idx]['set'] == 'A']
            Bi_partition = [all_items[idx] for idx in component if all_items[idx]['set'] == 'B']
            partitions.append((Ai_partition, Bi_partition))
            
    return partitions

def get_subsets(lst):
    """Generates all possible subsets (power set) of a given list."""
    subsets = []
    for r in range(len(lst) + 1):
        for comb in itertools.combinations(lst, r):
            subsets.append(list(comb))
    return subsets

def align_partition(Ai, Bi):
    """
    Tests all combinations of subsets in Ai and Bi, computes the score,
    and returns a tuple pair of index tuples that maximizes the score.
    """
    n_a = len(Ai)
    n_b = len(Bi)
    
    if n_a == 0 or n_b == 0:
        return []

    total_combinations = 2**(n_a + n_b)

    best_alignments = []
    
    if total_combinations < 10_000:
        # Do the exact search of the best alignment with exponential complexity
        subsets_a = get_subsets(Ai)
        subsets_b = get_subsets(Bi)
        
        best_score = float('-inf')
        best_subset_alignment = ()
        iter_count = 0
        for sa in subsets_a:
            for sb in subsets_b:
                iter_count += 1
                if iter_count % 1000 == 0:
                    #print(f"Processed {iter_count}/{total_combinations} subset pairs...")
                    pass
                boxes_a = [item['box'] for item in sa]
                boxes_b = [item['box'] for item in sb]
                
                iou_ab = set_iou(boxes_a, boxes_b)
                internal_a = internal_iou(boxes_a)
                internal_b = internal_iou(boxes_b)
                
                score = iou_ab - (internal_a + internal_b)
                
                if score > best_score:
                    best_score = score
                    best_subset_alignment = (sa, sb)
        best_alignments = [(a_bb['index'], b_bb['index']) for a_bb in best_subset_alignment[0] for b_bb in best_subset_alignment[1]]
    else:
        # For large partitions, use a greedy approach based on the Hungarian algorithm
        # Build pairwise IoU cost matrix (maximizing IoU -> minimizing negative IoU)
        cost_matrix = np.zeros((n_a, n_b))
        for r, item_a in enumerate(Ai):
            for c, item_b in enumerate(Bi):
                cost_matrix[r, c] = -set_iou([item_a['box']], [item_b['box']])
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        indices_a = []
        indices_b = []
        best_alignments = [(Ai[r]['index'], Bi[c]['index']) for r, c in zip(row_ind, col_ind) if -cost_matrix[r, c] > 0.0]
    return best_alignments                

def bounding_box_alignment(A, B):
    """
    Main pipeline returning a list of tuple pairs indexing the original sets A and B.
    """
    partitions = partition_boxes(A, B)
    alignments = []

    #print(f"Number of partitions: {len(partitions)}")
    
    for Ai, Bi in partitions:
        #print(f"Partition sizes: Ai={len(Ai)}, Bi={len(Bi)}")
        best_indices = align_partition(Ai, Bi)
        alignments.extend(best_indices)
        
    return alignments

def extract_bbs(data, src=True):
    svg_item_name = 'svgA' if src else 'svgB'
    for bb in data[svg_item_name]['boxes']:
        yield [bb['x'], bb['y'], bb['x'] + bb['width'], bb['y'] + bb['height']] 

# --- Example Usage ---
if __name__ == "__main__":
    #set_A = [[0, 0, 2, 2], [1, 1, 3, 3], [10, 10, 12, 12]]
    #set_B = [[0.5, 0.5, 2.5, 2.5], [11, 11, 13, 13]]

    data_manual = read_webapp(Path("manual/test"), "pair_053")
    orig_pair_id = get_orig_pair_id(data_manual)
    data_orig = read_orig(Path("data/test"), orig_pair_id)
    print(data_manual)
    print(orig_pair_id)
    print(data_orig)

    bbs_manual = list(extract_bbs(data_manual, False))
    bbs_orig = list(extract_bbs(data_orig, False))
    print(bbs_manual)
    print(bbs_orig)
    
    result_pairs = bounding_box_alignment(bbs_manual, bbs_orig)

    for manual_id, orig_id in result_pairs:
        print(f"({manual_id}, {orig_id}): \"{data_manual['svgB']['boxes'][manual_id]['text']}\"\t\"{data_orig['svgB']['boxes'][orig_id]['text']}\"")
