#!/usr/bin/env python3
"""
test_complexity.py — Test IDT/DLT/IDT+DLT/NND/LE on a single sentence.

Two modes:
  1. Raw text mode (requires Stanza):
     python test_complexity.py --text "The reporter who the senator attacked disliked the editor" --lang en

  2. Pre-parsed mode (no dependencies):
     python test_complexity.py --parsed "The/DET/2/det reporter/NOUN/7/nsubj who/PRON/6/obj the/DET/5/det senator/NOUN/6/nsubj attacked/VERB/2/acl:relcl disliked/VERB/0/root the/DET/9/det editor/NOUN/7/obj"

Pre-parsed format: token/UPOS/head/deprel separated by spaces.
Head is 1-based (0 = root).
"""

import sys, argparse
from collections import defaultdict


def build_sentence_from_parsed(parsed_str):
    """Parse 'token/UPOS/head/deprel token/UPOS/head/deprel ...' format."""
    sentence = []
    for i, item in enumerate(parsed_str.split()):
        parts = item.split('/')
        if len(parts) != 4:
            print(f"Error: '{item}' should be token/UPOS/head/deprel")
            sys.exit(1)
        sentence.append({
            'id': i + 1,
            'tok': parts[0],
            'upos': parts[1],
            'local_head': int(parts[2]),
            'deprel': parts[3],
        })
    return sentence


def build_sentence_from_stanza(text, lang):
    """Parse raw text with Stanza."""
    try:
        import stanza
    except ImportError:
        print("Error: Stanza not installed. Use --parsed mode or install stanza.")
        sys.exit(1)

    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse',
                              tokenize_pretokenized=False, verbose=False)
    except Exception:
        try:
            stanza.download(lang, verbose=False)
            nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse',
                                  tokenize_pretokenized=False, verbose=False)
        except Exception as e:
            print(f"Error loading Stanza model for '{lang}': {e}")
            sys.exit(1)

    doc = nlp(text)
    sentences = []
    for sent in doc.sentences:
        s = []
        for word in sent.words:
            s.append({
                'id': word.id,
                'tok': word.text,
                'upos': word.upos,
                'local_head': word.head,
                'deprel': word.deprel,
            })
        sentences.append(s)
    return sentences


# ============================================================
# Metrics (same as table_complexity.py)
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


def analyze_sentence(sentence):
    """Compute and display all metrics for a sentence."""
    n = len(sentence)
    links, links_compact = _build_links(sentence)

    # --- Tokens and dependency structure ---
    print(f"\n{'ID':>3s} {'Token':>12s} {'UPOS':>6s} {'Head':>5s} {'Deprel':>12s}")
    print("-" * 42)
    for tok in sentence:
        h_tok = sentence[tok['local_head']-1]['tok'] if tok['local_head'] > 0 else 'ROOT'
        print(f"{tok['id']:>3d} {tok['tok']:>12s} {tok['upos']:>6s} {tok['local_head']:>5d} {tok['deprel']:>12s}  <- {h_tok}")

    # --- IDT ---
    print(f"\n--- IDT (arc crossings) ---")
    idt_scores = []
    for i in range(n):
        backward = [l for l in links if (l[1][0] <= i and l[0][0] >= (i+1))]
        forward = [l for l in links if (l[0][0] <= i and l[1][0] >= (i+1))]
        count = len(backward) + len(forward)
        idt_scores.append(count)
        arcs = []
        for l in backward + forward:
            arcs.append(f"{l[0][1]}({l[0][0]+1})->{l[1][1]}({l[1][0]+1})")
        print(f"  {tok['id'] if False else i+1:>2d} {sentence[i]['tok']:>12s}  IDT={count:<3d} {', '.join(arcs) if arcs else '-'}")

    # --- DLT ---
    print(f"\n--- DLT (discourse referents on longest backward link) ---")
    ACCEPTABLE_POS = ['PROPN', 'NOUN', 'VERB']
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
                if links[indx][1][2] in ACCEPTABLE_POS:
                    accepted.append(f"{links[indx][1][1]}({links[indx][1][2]})")
        dlt_partial.append([link[0], len(accepted)])
        tok_name = sentence[link[0]]['tok']
        target_name = sentence[link[1]]['tok']
        print(f"  {tok_name}({link[0]+1}) -> {target_name}({link[1]+1}): referents={accepted} DLT={len(accepted)}")

    token_indices = [i for i,_ in dlt_partial]
    dlt_scores = []
    for i in range(n):
        if i not in token_indices: dlt_scores.append(0)
        else:
            vals = [p[1] for p in dlt_partial if p[0]==i]
            dlt_scores.append(vals[0] if vals else 0)

    # --- IDT+DLT ---
    idt_dlt_scores = [idt_scores[i] + dlt_scores[i] for i in range(n)]

    # --- NND ---
    print(f"\n--- NND (noun pair ancestor distance) ---")
    noun_ids = [tok['id'] for tok in sentence if tok['upos'] in ('NOUN', 'PROPN')]
    noun_toks = [tok['tok'] for tok in sentence if tok['upos'] in ('NOUN', 'PROPN')]
    print(f"  Nouns: {list(zip(noun_ids, noun_toks))}")

    parent = {tok['id']: tok['local_head'] for tok in sentence}
    def is_ancestor(a, b):
        current = b
        while current != 0:
            current = parent.get(current, 0)
            if current == a: return True
        return False

    nnd_scores = []
    if len(noun_ids) > 1:
        for i in range(len(noun_ids) - 1):
            fst, snd = noun_ids[i], noun_ids[i+1]
            ft, st = noun_toks[i], noun_toks[i+1]
            if is_ancestor(fst, snd):
                score = abs(fst - snd)
                print(f"  ({ft}, {st}): {ft} is ancestor of {st}, NND={score}")
            elif is_ancestor(snd, fst):
                score = abs(snd - fst)
                print(f"  ({ft}, {st}): {st} is ancestor of {ft}, NND={score}")
            else:
                score = 0
                print(f"  ({ft}, {st}): no ancestor relation, NND=0")
            nnd_scores.append(score)
    nnd_sum = sum(nnd_scores) if nnd_scores else 0

    # --- LE ---
    print(f"\n--- LE (tokens between verbs) ---")
    verb_indicator = []
    for tok in sentence:
        is_verb = tok['upos'] == 'VERB'
        is_aux_main = (tok['upos'] == 'AUX' and tok['deprel'] in ('root','cop'))
        verb_indicator.append(is_verb or is_aux_main)
    
    pattern = "".join("T" if v else "X" for v in verb_indicator)
    verbs = [tok['tok'] for tok in sentence if tok['upos'] == 'VERB' or (tok['upos'] == 'AUX' and tok['deprel'] in ('root','cop'))]
    print(f"  Verbs: {verbs}")
    print(f"  Pattern: {pattern}")

    if not any(verb_indicator):
        le_sum = 0
    else:
        parts = pattern.split("T")
        drop_last = True
        if parts[-1] == '': parts.pop(); drop_last = False
        if parts and parts[0] == '': parts.pop(0)
        counts = [len(p) for p in parts]
        if drop_last and counts: counts = counts[:-1]
        le_sum = sum(counts) if counts else 0
        print(f"  Gaps: {counts}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"{'ID':>3s} {'Token':>12s} {'IDT':>5s} {'DLT':>5s} {'IDT+DLT':>8s}")
    print("-" * 35)
    for i in range(n):
        print(f"{i+1:>3d} {sentence[i]['tok']:>12s} {idt_scores[i]:>5d} {dlt_scores[i]:>5d} {idt_dlt_scores[i]:>8d}")
    
    print(f"\n{'Metric':<12s} {'SUM':>6s} {'MEAN':>8s} {'MAX':>6s}")
    print("-" * 35)
    print(f"{'IDT':<12s} {sum(idt_scores):>6d} {sum(idt_scores)/n:>8.2f} {max(idt_scores):>6d}")
    print(f"{'DLT':<12s} {sum(dlt_scores):>6d} {sum(dlt_scores)/n:>8.2f} {max(dlt_scores):>6d}")
    print(f"{'IDT+DLT':<12s} {sum(idt_dlt_scores):>6d} {sum(idt_dlt_scores)/n:>8.2f} {max(idt_dlt_scores):>6d}")
    if nnd_scores:
        nnd_mean = sum(nnd_scores)/len(nnd_scores)
        nnd_max = max(nnd_scores)
        print(f"{'NND':<12s} {nnd_sum:>6d} {nnd_mean:>8.2f} {nnd_max:>6d}")
    else:
        print(f"{'NND':<12s} {nnd_sum:>6d} {0:>8.2f} {0:>6d}")
    if any(verb_indicator) and counts:
        le_mean = sum(counts)/len(counts)
        le_max = max(counts)
        print(f"{'LE':<12s} {le_sum:>6d} {le_mean:>8.2f} {le_max:>6d}")
    else:
        print(f"{'LE':<12s} {le_sum:>6d} {0:>8.2f} {0:>6d}")


def main():
    p = argparse.ArgumentParser(
        description='Test syntactic complexity metrics on a single sentence.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Raw text (requires Stanza)
  python test_complexity.py --text "The cat sat on the mat" --lang en

  # Pre-parsed (no dependencies needed)
  python test_complexity.py --parsed "The/DET/2/det reporter/NOUN/7/nsubj who/PRON/6/obj the/DET/5/det senator/NOUN/6/nsubj attacked/VERB/2/acl:relcl disliked/VERB/0/root the/DET/9/det editor/NOUN/7/obj"
""")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', '-t', help='Raw text to parse with Stanza')
    group.add_argument('--parsed', '-p', help='Pre-parsed: token/UPOS/head/deprel ...')
    p.add_argument('--lang', '-l', default='en', help='Language for Stanza (default: en)')
    args = p.parse_args()

    if args.parsed:
        sentence = build_sentence_from_parsed(args.parsed)
        print(f"Input: {' '.join(tok['tok'] for tok in sentence)}")
        analyze_sentence(sentence)
    else:
        sentences = build_sentence_from_stanza(args.text, args.lang)
        for i, sentence in enumerate(sentences):
            print(f"\n{'#'*60}")
            print(f"Sentence {i+1}: {' '.join(tok['tok'] for tok in sentence)}")
            analyze_sentence(sentence)


if __name__ == '__main__':
    main()
