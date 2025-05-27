import os
import json
import argparse
from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np
from glob import glob

from nltk.tokenize import sent_tokenize
import nltk
nltk.data.path.append("/data3/bubble3jh/nltk_data")

from retriever import obtain_relevant_evidences, get_wiki_from_db
from factuality_metric import nli_metric_batch, ner_metric
from src.claim_handling import obtain_important_ne, has_incorrect_style
from src.const import DATA_DIR, HOME_DIR, GEN_DIR

TYPES = {'NO_FACT': 1, 'HAS_FACT': 2, 'OFF_TOPIC': 3}

def identify_sentence_type(claim_obj, wiki_names_txt):
    if len(claim_obj['important_ne']) + len(claim_obj['unimportant_ne']) == 0 or has_incorrect_style(claim_obj) or len(claim_obj['subject']) == 0:
        return TYPES['NO_FACT']
    elif len(claim_obj['important_ne']) == 0 and len(claim_obj['unimportant_ne']) > 0:
        return TYPES['HAS_FACT']
    else:
        extra_ne = [ne[0] for ne in claim_obj['important_ne'] if ne[0] not in wiki_names_txt]
        overlap = claim_obj['subject'].intersection(set(" ".join(extra_ne).split(" ")))
        return TYPES['OFF_TOPIC'] if len(overlap) > 0 else TYPES['HAS_FACT']

def single_instance_eval(obj, prompt_wiki_names, run_nli_metric, run_ner_metric):
    wiki_names_txt = " ".join(prompt_wiki_names)
    text = obj['text'].strip()
    sents = sent_tokenize(text)
    if not sents:
        return {'claim_type': "NO_GEN"}

    gen_first_sent = sents[0]
    first_sent_obj_with_ne = obtain_important_ne(gen_first_sent.strip())
    sent_type = identify_sentence_type(first_sent_obj_with_ne, wiki_names_txt)
    claim_to_verify = first_sent_obj_with_ne['gen']

    result = {
        'claim_type': sent_type,
        'claim_to_verify': claim_to_verify,
        'hallu_ner': None,
        'nli-contr': None,
        'nli-entail': None,
        'nli-neutr': None,
        'nli-label': None,
        'used_ev': None,
        'evs': None,
        'top10': None
    }

    if sent_type != TYPES['HAS_FACT']:
        return result

    wiki_sentences = get_wiki_from_db(prompt_wiki_names)
    result['top10'] = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=5, method='combined')
    evs = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=1, method='combined')
    result['evs'] = evs

    if run_ner_metric:
        ne_to_check = first_sent_obj_with_ne['important_ne'] + first_sent_obj_with_ne['unimportant_ne']
        correct_ratio = ner_metric(ne_to_check, wiki_sentences)
        result['hallu_ner'] = 1 - correct_ratio

    if run_nli_metric:
        pairs = [[ev[0], claim_to_verify] for ev in evs]
        nli_probs, labels = nli_metric_batch(pairs)
        idx = np.argmax([p[2] for p in nli_probs])
        result['nli-contr'], result['nli-neutr'], result['nli-entail'] = nli_probs[idx]
        result['nli-label'] = labels[idx]
        result['used_ev'] = evs[idx]

    return result

def auto_resolve_gen_path(args):
    base_dir = "cp" if args.cp else "noncp"
    target_dir = os.path.join(GEN_DIR, "outputs", base_dir, args.prompt_type, args.model_name)
    merged_path = os.path.join(target_dir, "merged.jsonl")

    if not os.path.exists(merged_path):
        print(f"[INFO] Merged file not found at {merged_path}. Merging now…")
        os.makedirs(target_dir, exist_ok=True)

        jsonl_files = sorted(glob(os.path.join(target_dir, "*.jsonl")))
        with open(merged_path, "w") as outfile:
            for f in jsonl_files:
                with open(f) as infile:
                    for line in infile:
                        outfile.write(line)
        print(f"[INFO] Merged {len(jsonl_files)} files into {merged_path}")
    else:
        print(f"[INFO] Using existing merged file: {merged_path}")

    return merged_path


def main(args):
    run_nli_metric = True
    run_ner_metric = True

    # load prompts
    prompt_path = os.path.join(HOME_DIR, "prompts", f"fever_{args.prompt_type}_final.jsonl")
    with open(prompt_path) as f:
        prompts = [json.loads(line.strip()) for line in f]

    # resolve generation path
    if args.gen_path is not None:
        gen_path = os.path.join(GEN_DIR, args.gen_path)
    else:
        gen_path = auto_resolve_gen_path(args)

    with open(gen_path) as f:
        gens = [json.loads(line.strip()) for line in f]

    if args.debug_sample_size:
        prompts = prompts[:args.debug_sample_size]
        gens = gens[:args.debug_sample_size]
    import pdb; pdb.set_trace()
    assert len(prompts) == len(gens), "Mismatch between prompt and gen length"

    final_results = []
    final_hallu_ner_score = final_contradict_prob = final_neutral_prob = final_entail_prob = 0
    strict_true_ner_score = 0
    no_fact_cnt = has_fact_cnt = off_topic_cnt = 0
    all_nli_labels = []

    for prompt_obj, gen_obj in tqdm(zip(prompts, gens), total=len(prompts)):
        prompt_wiki_names = [e[0] for e in prompt_obj['evidence_info']]
        res = single_instance_eval(gen_obj, prompt_wiki_names, run_nli_metric, run_ner_metric)

        if res['claim_type'] == TYPES['NO_FACT']:
            no_fact_cnt += 1
        elif res['claim_type'] == TYPES['OFF_TOPIC']:
            off_topic_cnt += 1
        elif res['claim_type'] == TYPES['HAS_FACT']:
            has_fact_cnt += 1
            final_hallu_ner_score += res['hallu_ner']
            final_contradict_prob += res['nli-contr']
            final_neutral_prob += res['nli-neutr']
            final_entail_prob += res['nli-entail']
            all_nli_labels.append(res['nli-label'])

            if res['hallu_ner'] == 1.0 and res['nli-label'] == 2:
                strict_true_ner_score += 1

            if args.save_gen_for_analysis:
                final_results.append({
                    'wiki': " ".join(prompt_wiki_names),
                    'prompt': gen_obj['prompt'],
                    'lm-gen': res['claim_to_verify'],
                    'hallu_ner': res['hallu_ner'],
                    'nli-label': res['nli-label'],
                    'nli-entail': res['nli-entail'],
                    'nli-contr': res['nli-contr'],
                    'nli-neutral': res['nli-neutr'],
                    'used_ev': res['used_ev'],
                    'top10': res['top10']
                })

    total = no_fact_cnt + has_fact_cnt + off_topic_cnt
    print(f"NO_FACT: {no_fact_cnt/total:.2%}, HAS_FACT: {has_fact_cnt/total:.2%}, OFF_TOPIC: {off_topic_cnt/total:.2%}")
    print(f"Hallu NER: {final_hallu_ner_score/has_fact_cnt:.2%}, Strict True NER: {strict_true_ner_score/has_fact_cnt:.2%}")
    print(f"AVG PROBS: Contradict: {final_contradict_prob/has_fact_cnt:.2%}, Neutral: {final_neutral_prob/has_fact_cnt:.2%}, Entail: {final_entail_prob/has_fact_cnt:.2%}")

    if run_nli_metric:
        c = Counter(all_nli_labels)
        total_nli = sum(c.values())
        print(f"NLI CLASS %: Contradict: {c[0]/total_nli:.2%}, Neutral: {c[1]/total_nli:.2%}, Entail: {c[2]/total_nli:.2%}")

    if args.save_gen_for_analysis:
        out_csv = gen_path.replace(".jsonl", "_analysis.csv")
        pd.DataFrame(final_results).to_csv(out_csv)
        print(f"[SAVED] analysis to {out_csv}")

    # 결과 저장
    res_path = gen_path.replace(".jsonl", "_results.jsonl")
    with open(res_path, 'a') as outfile:
        json.dump({
            "avg_hallu_ner_ratio": final_hallu_ner_score / has_fact_cnt,
            "nli_contradict_class_ratio": c[0]/total_nli,
            "nli_neutral_class_ratio": c[1]/total_nli,
            "nli_entail_class_ratio": c[2]/total_nli,
            "no_fact_ratio": no_fact_cnt/total,
            "has_fact_ratio": has_fact_cnt/total,
            "off_topic_ratio": off_topic_cnt/total
        }, outfile)
        outfile.write("\n")

# run command: PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type nonfactual --model_name qwen3_1.7b
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', type=str, required=True, choices=['factual', 'nonfactual'])
    parser.add_argument('--model_name', type=str, default="qwen3_0.6b")
    parser.add_argument('--cp', action='store_true')
    parser.add_argument('--gen_path', type=str, default=None)
    parser.add_argument('--debug_sample_size', type=int, default=None)
    parser.add_argument('--save_gen_for_analysis', action='store_true')
    args = parser.parse_args()
    main(args)
