from argparse import ArgumentParser
import os
import torch
import pdb
import json

from transformers import AutoTokenizer

from main_loop import main_loop, prediction, main_parser
from models import QAReasoner
from hotpotqa.hotpot_evaluate_utils import eval_func

def logits2span(start_logits, end_logits, top_k=5):
    top_start_logits, top_start_indices = torch.topk(start_logits.squeeze_(0), k=top_k)
    top_end_logits, top_end_indices = torch.topk(end_logits.squeeze_(0), k=top_k)
    ret = []
    for start_pos in top_start_indices:
        for end_pos in top_end_indices:
            if end_pos - start_pos < 0:
                adds = -100000
            elif end_pos - start_pos > 20:
                adds = -20
            elif end_pos - start_pos > 8:
                adds = -5
            else:
                adds = 0
            ret.append((adds + start_logits[start_pos] + end_logits[end_pos], start_pos, end_pos))
    ret.sort(reverse=True)
    return ret[0][1], ret[0][2] + 1

def extract_supporing_facts(config, buf, score, start, end):
    ret = []
    # the result sentence
    for i, sen_end in enumerate(buf.block_ends()):
        if sen_end >= end:
            if buf[i].blk_type > 0:
                ret.append(list(buf[i].origin))
            break
    # best 2 entity sentence
    for idx in score.argsort(descending=True):
        if buf[idx].blk_type > 0:
            entity = buf[idx].origin[0]
            if all([entity != fact[0] for fact in ret]):
                ret.append(list(buf[idx].origin))
            if len(ret) >= 2:
                break
    # auxiliary sp
    gold_entities = [t[0] for t in ret if t[1] != 0]
    for i, blk in enumerate(buf):
        if buf[i].blk_type > 0:
            entity, sen_idx = blk.origin
            if entity in gold_entities and score[i] + 0.05 * int(sen_idx == 0) > config.sp_threshold and [entity, sen_idx] not in ret:
                ret.append([entity, sen_idx])
    return ret

if __name__ == "__main__":
    print('Please confirm the hotpotqa data are ready by ./hotpotqa/process_hotpotqa.py!')
    print('=====================================')
    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = ArgumentParser(add_help=False)
    # ------------ add hotpotqa argument ----------
    parser.add_argument('--sp_threshold', type=int, default=0.9)
    parser.add_argument('--only_predict', action='store_true')
    # ---------------------------------------------
    parser = main_parser(parser)
    parser.set_defaults(
        train_source = os.path.join(root_dir, 'data', 'hotpotqa_train_roberta-base.pkl'),
        test_source = os.path.join(root_dir, 'data', 'hotpotqa_test_roberta-base.pkl'),
        introspect = True
    )
    config = parser.parse_args()
    config.reasoner_cls_name = 'QAReasoner'
    if not config.only_predict: # train
        main_loop(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    sp, ans = {}, {}
    for qbuf, dbuf, buf, relevance_score, ids, output in prediction(config):
        _id = qbuf[0]._id
        start, end = logits2span(*output)
        ans_ids = ids[start: end]
        ans[_id] = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ans_ids)).replace('</s>', '').replace('<pad>', '').strip()
        # supporting facts  
        sp[_id] = extract_supporing_facts(config, buf, relevance_score, start, end)
    with open(os.path.join(config.tmp_dir, 'pred.json'), 'w') as fout:
        pred = {'answer': ans, 'sp': sp}
        json.dump(pred, fout)
        print(eval_func(pred, os.path.join(root_dir, 'data', 'hotpot_dev_distractor_v1.json')))

