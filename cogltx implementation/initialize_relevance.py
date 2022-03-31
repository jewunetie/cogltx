import gensim.downloader as api
import re
import numpy as np
from tqdm import tqdm

def remove_special_split(blk):
    return re.sub(r'</s>|<pad>|<s>|\W', ' ', str(blk)).lower().split()

def _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transforms = None, threshold=0.15):
    if conditional_transforms is None:
        conditional_transforms = []
    for transform_func in conditional_transforms:
        qbuf, dbuf = transform_func(qbuf, dbuf)
    dvecs = []
    for blk in dbuf:
        if doc := [
            word_vectors[w]
            for w in remove_special_split(blk)
            if w in word_vectors
        ]:
            dvecs.append(np.stack(doc))
        else:
            dvecs.append(np.zeros((1, 100)))
    
    qvec = np.stack([word_vectors[w] for w in remove_special_split(qbuf) if w in word_vectors])
    scores = [np.matmul(qvec, dvec.T).mean() for dvec in dvecs]
    max_score_abs = max(scores) - min(scores) + 1e-6
    
    for i, blk in enumerate(dbuf):
        if 1 - scores[i] / max_score_abs < threshold:
            blk.relevance = max(blk.relevance, 1)
    return True


def init_relevance(a, method='glove', conditional_transforms = None):
    if conditional_transforms is None:
        conditional_transforms = []
    print('Initialize relevance...')
    total = 0
    if method == 'glove':
        word_vectors = api.load("glove-wiki-gigaword-100")
        for qbuf, dbuf in tqdm(a):
            total += _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transforms)
    print(f'Initialized {total} question-document pairs!')