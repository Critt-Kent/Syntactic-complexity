#!/usr/bin/env python3
"""
table_complexity.py — Compute syntactic complexity metrics from TPR-DB table files (.st, .tt).

Metrics:
  Token-level: IDT, DLT, IDT+DLT
  Segment-level: IDT, DLT, IDT+DLT, NND, LE, BCR

Usage:
    python table_complexity.py P01_Plt1.st                          # ST segment-level, sum
    python table_complexity.py P01_Plt1.tt                          # TT segment-level, sum
    python table_complexity.py P01_Plt1.st P01_Plt1.tt              # ST + TT + BCR
    python table_complexity.py P01_Plt1.st -f token                 # per-token
    python table_complexity.py P01_Plt1.st -a mean                  # mean aggregation
    python table_complexity.py P01_Plt1.st P01_Plt1.tt -o results.csv

Requirements: Python 3.8+, pandas

Reference: Zou, L. (2024). Cognitive Processes in Human-ChatGPT Interaction during Machine Translation Post-editing.
"""

import pandas as pd
import csv, sys, os, argparse
from collections import defaultdict


# ============================================================
# Read and convert tables to sentence format
# ============================================================

def read_table(filepath):
    """Read a TPR-DB .st or .tt table file."""
    df = pd.read_csv(filepath, sep='\t', dtype=None)
    return df


def table_to_sentences(df, token_col, id_col, seg_col):
    """
    Convert a TPR-DB table to sentence dicts grouped by segment.
    
    Args:
        df: DataFrame with token data
        token_col: 'SToken' or 'TToken'
        id_col: 'STid' or 'TTid'
        seg_col: 'STseg' or 'TTseg'
    
    Returns:
        dict: {seg_id: [token_dict, ...]}
    """
    sentences = {}
    
    for seg_id, group in df.groupby(seg_col):
        group = group.sort_values(id_col).reset_index(drop=True)
        offset = group[id_col].min() - 1
        
        sentence = []
        for _, row in group.iterrows():
            local_id = int(row[id_col]) - offset
            local_head = int(row['head']) - offset if pd.notna(row['head']) else 0
            # head == 0 means root
            if local_head < 0 or local_head > len(group):
                local_head = 0
            
            sentence.append({
                'id': local_id,
                'global_id': int(row[id_col]),
                'tok': str(row[token_col]),
                'upos': str(row.get('upos', '')),
                'xpos': str(row.get('xpos', '')),
                'head': int(row['head']) if pd.notna(row['head']) else 0,
                'local_head': local_head,
                'deprel': str(row.get('deprel', '')),
            })
        
        sentences[str(int(seg_id))] = sentence
    
    return sentences


def detect_table_type(df):
    """Detect if this is an ST or TT table."""
    if 'SToken' in df.columns:
        return 'ST', 'SToken', 'STid', 'STseg'
    elif 'TToken' in df.columns:
        return 'TT', 'TToken', 'TTid', 'TTseg'
    else:
        raise ValueError("Cannot detect table type: no SToken or TToken column found")


# ============================================================
# Link building
# ============================================================

def _build_links(sentence):
    links = []; links_compact = []
    for tok in sentence:
        if tok['local_head'] > 0:
            h = sentence[tok['local_head']-1]
            links.append([[tok['local_head']-1, h['tok'], h['upos']],
                         [tok['id']-1, tok['tok'], tok['upos']]])
            links_compact.append([tok['local_head']-1, tok['id']-1])
        else:
            links.append([[tok['id']-1, tok['tok'], tok['upos']],
                         [tok['id']-1, tok['tok'], tok['upos']]])
            links_compact.append([tok['id']-1, tok['id']-1])
    return links, links_compact


# ============================================================
# Per-token metrics
# ============================================================

def compute_idt_per_token(sentence):
    links, links_compact = _build_links(sentence)
    scores = []
    for i in range(len(links_compact)):
        backward = [l for l in links if (l[1][0] <= i and l[0][0] >= (i+1))]
        forward = [l for l in links if (l[0][0] <= i and l[1][0] >= (i+1))]
        scores.append(len(backward) + len(forward))
    return scores


def compute_dlt_per_token(sentence):
    ACCEPTABLE_POS = ['PROPN', 'NOUN', 'VERB']
    links, links_compact = _build_links(sentence)
    backward_longest = []
    for link in links:
        dep_idx = link[1][0]
        for indx in range(0, dep_idx):
            if [dep_idx, indx] in links_compact:
                backward_longest.append([dep_idx, indx]); break
    dlt_partial = []
    for link in backward_longest:
        accepted = []
        for indx in range(link[0]-1, link[1]-1, -1):
            if 0 <= indx < len(links):
                if links[indx][1][2] in ACCEPTABLE_POS: accepted.append(1)
        dlt_partial.append([link[0], len(accepted)])
    token_indices = [i for i,_ in dlt_partial]
    scores = []
    for i in range(len(links_compact)):
        if i not in token_indices: scores.append(0)
        else:
            vals = [p[1] for p in dlt_partial if p[0]==i]
            scores.append(vals[0] if vals else 0)
    return scores


def compute_idt_dlt_per_token(sentence):
    idt = compute_idt_per_token(sentence)
    dlt = compute_dlt_per_token(sentence)
    return [idt[i] + dlt[i] for i in range(len(idt))]


# ============================================================
# Aggregation
# ============================================================

def _aggregate(scores, aggregation='sum'):
    if not scores: return 0
    if aggregation == 'sum': return sum(scores)
    elif aggregation == 'mean': return sum(scores)/len(scores) if scores else 0
    elif aggregation == 'max': return max(scores) if scores else 0
    return sum(scores)


# ============================================================
# Segment-level metrics
# ============================================================

def compute_idt(sentence, aggregation='sum'):
    return _aggregate(compute_idt_per_token(sentence), aggregation)

def compute_dlt(sentence, aggregation='sum'):
    return _aggregate(compute_dlt_per_token(sentence), aggregation)

def compute_idt_dlt(sentence, aggregation='sum'):
    return _aggregate(compute_idt_dlt_per_token(sentence), aggregation)


def compute_nnd(sentence, aggregation='sum'):
    noun_ids = [tok['id'] for tok in sentence if tok['upos'] in ('NOUN', 'PROPN')]
    if len(noun_ids) <= 1: return 0
    parent = {tok['id']: tok['local_head'] for tok in sentence}
    def is_ancestor(a, b):
        current = b
        while current != 0:
            current = parent.get(current, 0)
            if current == a: return True
        return False
    scores = []
    for i in range(len(noun_ids) - 1):
        fst, snd = noun_ids[i], noun_ids[i+1]
        if is_ancestor(fst, snd): scores.append(abs(fst - snd))
        elif is_ancestor(snd, fst): scores.append(abs(snd - fst))
        else: scores.append(0)
    return _aggregate(scores, aggregation)


def compute_le(sentence, aggregation='sum'):
    verb_indicator = []
    for tok in sentence:
        is_verb = tok['upos'] == 'VERB'
        is_aux_main = (tok['upos'] == 'AUX' and tok['deprel'] in ('root','cop'))
        verb_indicator.append(is_verb or is_aux_main)
    if not any(verb_indicator): return 0
    s = "".join("T" if v else "X" for v in verb_indicator)
    parts = s.split("T")
    drop_last = True
    if parts[-1] == '': parts.pop(); drop_last = False
    if parts and parts[0] == '': parts.pop(0)
    counts = [len(p) for p in parts]
    if drop_last and counts: counts = counts[:-1]
    return _aggregate(counts, aggregation)


def sent_len(sentence, aggregation='sum'):
    return sum(1 for t in sentence if t['upos'] != 'PUNCT')

def compute_bcr(st_val, tt_val):
    return tt_val / st_val if st_val > 0 else 0


METRICS = [
    ('len', 'Tokens', sent_len),
    ('idt', 'IDT', compute_idt),
    ('dlt', 'DLT', compute_dlt),
    ('idt_dlt', 'IDT+DLT', compute_idt_dlt),
    ('nnd', 'NND', compute_nnd),
    ('le', 'LE', compute_le),
]


# ============================================================
# Processing
# ============================================================

def process_table(st_file, tt_file=None, aggregation='sum'):
    """Process TPR-DB table file(s) and compute all metrics."""
    
    # Read ST
    st_df = read_table(st_file)
    st_type, st_tok_col, st_id_col, st_seg_col = detect_table_type(st_df)
    st_sents = table_to_sentences(st_df, st_tok_col, st_id_col, st_seg_col)
    
    # Read TT if provided
    tt_sents = {}
    if tt_file:
        tt_df = read_table(tt_file)
        tt_type, tt_tok_col, tt_id_col, tt_seg_col = detect_table_type(tt_df)
        tt_sents = table_to_sentences(tt_df, tt_tok_col, tt_id_col, tt_seg_col)
    
    # Per-token results
    token_results = []
    for text_type, sents, tok_col in [('ST', st_sents, st_tok_col)]:
        for seg in sorted(sents.keys(), key=lambda x: int(x)):
            sentence = sents[seg]
            idt_scores = compute_idt_per_token(sentence)
            dlt_scores = compute_dlt_per_token(sentence)
            idt_dlt_scores = compute_idt_dlt_per_token(sentence)
            for i, tok in enumerate(sentence):
                token_results.append({
                    'seg': seg, 'type': text_type,
                    'token_id': tok['global_id'], 'token': tok['tok'],
                    'upos': tok['upos'], 'head': tok['head'], 'deprel': tok['deprel'],
                    'idt': idt_scores[i], 'dlt': dlt_scores[i], 'idt_dlt': idt_dlt_scores[i],
                })
    
    if tt_sents:
        for seg in sorted(tt_sents.keys(), key=lambda x: int(x)):
            sentence = tt_sents[seg]
            idt_scores = compute_idt_per_token(sentence)
            dlt_scores = compute_dlt_per_token(sentence)
            idt_dlt_scores = compute_idt_dlt_per_token(sentence)
            for i, tok in enumerate(sentence):
                token_results.append({
                    'seg': seg, 'type': 'TT',
                    'token_id': tok['global_id'], 'token': tok['tok'],
                    'upos': tok['upos'], 'head': tok['head'], 'deprel': tok['deprel'],
                    'idt': idt_scores[i], 'dlt': dlt_scores[i], 'idt_dlt': idt_dlt_scores[i],
                })
    
    # Per-segment results
    seg_results = {'segments': []}
    all_segs = sorted(set(list(st_sents.keys()) + list(tt_sents.keys())), key=lambda x: int(x))
    
    for seg in all_segs:
        row_st = row_tt = row_bcr = None
        
        if seg in st_sents:
            row_st = {'seg': seg, 'type': 'ST'}
            for key, name, fn in METRICS:
                row_st[key] = fn(st_sents[seg], aggregation)
        
        if seg in tt_sents:
            row_tt = {'seg': seg, 'type': 'TT'}
            for key, name, fn in METRICS:
                row_tt[key] = fn(tt_sents[seg], aggregation)
            
            if row_st:
                row_bcr = {'seg': seg, 'type': 'BCR'}
                for key, name, fn in METRICS:
                    if key != 'len':
                        row_bcr[key] = compute_bcr(row_st[key], row_tt[key])
        
        seg_results['segments'].append({'st': row_st, 'tt': row_tt, 'bcr': row_bcr})
    
    return token_results, seg_results


def print_token_results(token_results):
    print(f"{'Seg':>4s} {'Type':>4s} {'TokID':>6s} {'Token':>10s} {'UPOS':>6s} {'IDT':>5s} {'DLT':>5s} {'IDT+DLT':>8s}")
    print("-" * 55)
    for r in token_results:
        print(f"{r['seg']:>4s} {r['type']:>4s} {r['token_id']:>6d} {r['token']:>10s} {r['upos']:>6s} {r['idt']:>5d} {r['dlt']:>5d} {r['idt_dlt']:>8d}")


def print_segment_results(seg_results, aggregation='sum'):
    is_int = aggregation in ('sum', 'max')
    
    has_tt = any(s['tt'] for s in seg_results['segments'])
    
    h = f"{'Seg':>4s} {'Type':>4s} {'Len':>5s} {'IDT':>6s} {'DLT':>6s} {'IDT+DLT':>8s} {'NND':>5s} {'LE':>4s}"
    print(f"Aggregation: {aggregation}")
    print(h); print("-"*len(h))
    for sd in seg_results['segments']:
        for tt in ['st', 'tt']:
            r = sd[tt]
            if r:
                if is_int:
                    print(f"{r['seg']:>4s} {r['type']:>4s} {r['len']:>5d} {r['idt']:>6d} {r['dlt']:>6d} {r['idt_dlt']:>8d} {r['nnd']:>5d} {r['le']:>4d}")
                else:
                    print(f"{r['seg']:>4s} {r['type']:>4s} {r['len']:>5d} {r['idt']:>6.2f} {r['dlt']:>6.2f} {r['idt_dlt']:>8.2f} {r['nnd']:>5.2f} {r['le']:>4.2f}")
    
    if has_tt:
        print()
        h2 = f"{'Seg':>4s} {'BCR_IDT':>8s} {'BCR_DLT':>8s} {'BCR_IDT+DLT':>12s} {'BCR_NND':>8s} {'BCR_LE':>8s}"
        print(h2); print("-"*len(h2))
        for sd in seg_results['segments']:
            b = sd['bcr']
            if b:
                print(f"{b['seg']:>4s} {b.get('idt',0):>8.3f} {b.get('dlt',0):>8.3f} {b.get('idt_dlt',0):>12.3f} {b.get('nnd',0):>8.3f} {b.get('le',0):>8.3f}")


def write_token_csv(token_results, filepath):
    fns = ['seg','type','token_id','token','upos','head','deprel','idt','dlt','idt_dlt']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in token_results: w.writerow(r)
    print(f"Token results saved to {filepath}")


def write_segment_csv(seg_results, filepath):
    rows = []
    for sd in seg_results['segments']:
        for tt in ['st', 'tt', 'bcr']:
            if sd[tt]: rows.append(sd[tt])
    if not rows: return
    fns = ['seg', 'type'] + [k for k,_,_ in METRICS]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[k for k in fns if k in rows[0]], extrasaction='ignore')
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Segment results saved to {filepath}")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description='Compute syntactic complexity from TPR-DB table files (.st, .tt).',
        epilog="""Examples:
  python table_complexity.py P01_Plt1.st                     # ST only
  python table_complexity.py P01_Plt1.st P01_Plt1.tt         # ST + TT + BCR
  python table_complexity.py P01_Plt1.st -f token            # per-token
  python table_complexity.py P01_Plt1.st P01_Plt1.tt -a mean # mean aggregation
  python table_complexity.py P01_Plt1.st -o results.csv      # CSV export
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('st_file', help='ST or TT table file (.st or .tt)')
    p.add_argument('tt_file', nargs='?', default=None, help='TT table file (optional, for BCR)')
    p.add_argument('--output', '-o', help='Output CSV file')
    p.add_argument('--format', '-f', choices=['token', 'segment'], default='segment')
    p.add_argument('--aggregation', '-a', choices=['sum', 'mean', 'max'], default='sum')
    args = p.parse_args()
    
    if not os.path.exists(args.st_file):
        print(f"Error: {args.st_file} not found", file=sys.stderr); sys.exit(1)
    if args.tt_file and not os.path.exists(args.tt_file):
        print(f"Error: {args.tt_file} not found", file=sys.stderr); sys.exit(1)
    
    token_results, seg_results = process_table(args.st_file, args.tt_file, args.aggregation)
    
    if args.format == 'token':
        print_token_results(token_results)
        if args.output: write_token_csv(token_results, args.output)
    else:
        print_segment_results(seg_results, args.aggregation)
        if args.output: write_segment_csv(seg_results, args.output)


if __name__ == '__main__':
    main()
