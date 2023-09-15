import concurrent.futures
import itertools
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from time import time

import numpy as np
import sklearn.metrics
from loguru import logger
from sklearn.metrics import ndcg_score
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from tqdm import tqdm

from shared.utility import is_debug_mode


def test(path, mapped, scores, k):
    item_matrix = np.load(os.path.join(path, 'items.npy'), mmap_mode='r')
    label_matrix = np.load(os.path.join(path, 'labels.npy'), mmap_mode='r')

    items = item_matrix[mapped]
    true_relevance = label_matrix[mapped]

    scores = np.take_along_axis(scores, items, axis=-1)  # Sort scores according to items

    indices = np.argpartition(scores, kth=-k, axis=-1)[:, -k:][:, ::-1]

    true_relevance = np.take_along_axis(true_relevance, indices, axis=-1)
    scores = np.take_along_axis(scores, indices, axis=-1)

    # Efficient sorting of scores. Only sort last k, in increasing order, therefore reverse.
    ndcgs = _ndcg_sample_scores(true_relevance, scores, ignore_ties=True)

    del items, true_relevance, scores, indices

    return np.sum(ndcgs)


class Validator:
    def __init__(self, path, batch_size, full=True):
        self.batch_size = batch_size
        self.path = path
        self.full = full

        self.item_matrix = np.load(os.path.join(path, 'items.npy'), mmap_mode='r')
        self.label_matrix = np.load(os.path.join(path, 'labels.npy'), mmap_mode='r')
        self.val_users = {user.index: (idx, [i for i, r in user.loo], [i for i, r in user.ratings]) for idx, user in
                                       enumerate(sorted(pickle.load(open(os.path.join(path, 'validation.pickle'), 'rb'))
                                                        , key=lambda x: x.index))}

        info = pickle.load(open(os.path.join(path, 'info.pickle'), 'rb'))
        self.k = info['k']
        self.idcg = info['idcg']
        self.users = info['users']
        self.user_mapping = {user: idx for idx, user in enumerate(self.users)}

    def dcg(self, model, mapper, batch, max_validation):
        mapped = mapper(batch)

        items = self.item_matrix[mapped]
        true_relevance = self.label_matrix[mapped]

        if max_validation:
            n, m = items.shape
            pos = np.where(true_relevance)

            n_pairs = len(pos[0])
            pos_pairs = np.stack(pos).T[np.random.choice(n_pairs, n_pairs, replace=False)]  # shuffle order
            _, selected = np.unique(pos_pairs[:, 0], return_index=True)  # return first positive rating for each user

            # Set probability of positive pairs to zero
            permutation = np.zeros((n, max_validation - 1))
            for u in np.arange(n):
                p = np.ones(m)
                p[np.where(true_relevance[u])[0]] = 0  # Set positive to false
                permutation[u] = np.random.choice(m, max_validation-1, p=p/sum(p))
            permutation = permutation.astype(np.int)

            pos_items = items[np.arange(n), pos_pairs[selected, 1]]
            items = np.take_along_axis(items, permutation, axis=-1)
            true_relevance = np.take_along_axis(true_relevance, permutation, axis=-1)

            items = np.hstack((pos_items.reshape(-1,1), items))
            true_relevance = np.hstack((np.ones((n, 1)), true_relevance))

            scores = model.predict_all(batch, items)

            dcg = _ndcg_sample_scores(true_relevance, scores, ignore_ties=True)
        else:
            scores = model.predict_all(batch)

            scores = np.take_along_axis(scores, items, axis=-1)  # only use scores for items in item matrix.

            indices = np.argpartition(scores, kth=-self.k, axis=-1)[:, -self.k:][:, ::-1]

            true_relevance = np.take_along_axis(true_relevance, indices, axis=-1)
            scores = np.take_along_axis(scores, indices, axis=-1)

            # Efficient sorting of scores. Only sort last k, in increasing order, therefore reverse.
            dcg = _dcg_sample_scores(true_relevance, scores, ignore_ties=True)

            # Use for debug. If uncommented, should return 1.
            # indices = np.argpartition(true_relevance, kth=-self.k, axis=-1)[:, -self.k:][:, ::-1]
            # true_relevance = np.take_along_axis(true_relevance, indices, axis=-1)
            # dcg = _dcg_sample_scores(true_relevance, true_relevance, ignore_ties=True)
            # logger.warning('calculating IDCG')

        ndcg = dcg / self.idcg[mapped]
        # assert np.allclose(ndcg, np.ones_like(ndcg))
        return np.sum(ndcg)

    def threaded(self, model, batches, mapper):
        ndcg = 0
        futures =[]
        nworkers = 2
        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            for batch in tqdm(batches):
                if len(futures) >= nworkers:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for idx, f in filter(lambda x: x[1].done(), enumerate(futures)):
                        futures.pop(idx)
                        ndcg += f.result()
                scores = model.predict_all(batch)
                mapped = mapper(batch)
                futures.append(executor.submit(test, self.path, mapped, scores, self.k))

            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                ndcg += f.result()

    def _sub_validate(self, model, workers=5, threaded=True, max_validation=None, max_users=None, verbose=True, test_users=None):
        # Get number of batches and map users to indices
        users = self.users if test_users is None else test_users
        n_batches = (len(users) // self.batch_size) + 1
        mapper = np.vectorize(self.user_mapping.get)

        if max_users is not None:
            assert max_validation is not None, 'If using max users, ensure max validation or might be wrong.'

            users = np.random.permutation(users)
            users = users[:max_users]

        batches = [users[b * self.batch_size:(b + 1) * self.batch_size] for b in range(n_batches)]

        # ndcg = self.threaded(model, batches, mapper)

        ndcg = 0
        for b in tqdm(batches, desc='Validating', disable=not is_debug_mode() or not verbose):
            if len(b):
                ndcg += self.dcg(model, mapper, b, max_validation)

        ndcg = ndcg / len(users)

        # if max_validation is None:
        #     ndcg = dcg / self.idcg
        # else:
        #     ndcg = dcg

        return ndcg

    def _full_validate(self, model):
        progress = tqdm(total=3, desc='Validating', disable=not is_debug_mode())

        rec_items = []
        user_iterator = iter(self.val_users)

        for i in range(0, len(self.val_users), self.batch_size):
            batch = list(itertools.islice(user_iterator, self.batch_size))
            scores = model.predict_all(batch)
            for i, user in enumerate(batch):
                scores[i, self.val_users[user][2]] = -np.inf

            items = np.argpartition(scores, -self.k)[:,-self.k:]
            items = np.take_along_axis(items, np.argsort(np.take_along_axis(scores, items, axis=-1)), axis=-1)[:,::-1]
            rec_items.append(items)

        progress.update()

        rec_items = np.concatenate(rec_items, axis=0)
        hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
        for i, (user, (idx, test_items, _)) in enumerate(self.val_users.items()):
            for item_idx in range(rec_items.shape[1]):
                if rec_items[idx, item_idx] in test_items:
                    hit_matrix[idx, item_idx] = 1.
        eval_data_len = np.array([len(test_items) for _, test_items, _ in self.val_users.values()], dtype=np.int32)

        progress.update()

        max_hit_num = np.minimum(eval_data_len, self.k)
        max_hit_matrix = np.zeros_like(hit_matrix[:, :self.k], dtype=np.float32)
        for user, num in enumerate(max_hit_num):
            max_hit_matrix[user, :num] = 1.
        denominator = np.log2(np.arange(2, self.k + 2, dtype=np.float32))[None, :]
        dcgs = np.sum(hit_matrix[:, :self.k] / denominator, axis=1)
        idcgs = np.sum(max_hit_matrix / denominator, axis=1)
        with np.errstate(invalid='ignore'):
            ndcgs = dcgs / idcgs

        progress.update()
        progress.close()

        return ndcgs.mean()

    def validate(self, model, workers=5, threaded=True, max_validation=None, max_users=None, verbose=True, test_users=None):
        if self.full:
            return self._full_validate(model)
        else:
            return self._sub_validate(model, workers, threaded, max_validation, max_users, verbose, test_users)


if __name__ == '__main__':
    dataset = '/srv/data/tjendal/Projects/cold-start-framework/datasets/ml-mr/ml_mr_warm_start'
    bs = 512
    v = Validator(os.path.join(dataset, 'fold_0'), bs)
    meta = pickle.load(open(os.path.join(dataset, 'meta.pickle'), 'rb'))

    def foo(x):
        return np.random.rand(len(x), len(meta.items))

    t = time()
    res = v.validate(foo, workers=8)
    t = time() - t
    print(f'ndcg: {res}, t:{t}')
