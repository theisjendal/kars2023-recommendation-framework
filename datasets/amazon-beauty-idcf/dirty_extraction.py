import pickle
import random
from collections import defaultdict

import numpy as np
import os

from shared.entity import Entity
from shared.enums import Sentiment
from shared.meta import Meta
from shared.user import User, LeaveOneOutUser
from shared.utility import save_pickle

datadir = 'beauty_s20.pkl'

try:
    with open(datadir, 'rb') as f:
        ucs_set = pickle.load(f)
        cs_set = pickle.load(f)
        u_his_list = pickle.load(f)
        i_his_list = pickle.load(f)
        ucs_count, cs_count, item_count = pickle.load(f)
except:
    with open(datadir, 'rb') as f:
        ucs_set = pickle.load(f)
        cs_set = pickle.load(f)
        ucs_count, cs_count, item_count = pickle.load(f)

train_set_supp, test_set_supp = [], []
train_set_que, test_set_que = [], []

for u in range(len(ucs_set)):
    train_set_supp += ucs_set[u][:-10]
    test_set_supp += ucs_set[u][-10:]

for u in range(len(cs_set)):
    train_set_que += cs_set[u][:-10]
    test_set_que += cs_set[u][-10:]

random.seed(42)
np.random.seed(42)

train_set = np.array(train_set_supp)
test_set = np.array(test_set_que)  # For extrapolation
items = set(np.concatenate([train_set[:, 1], test_set[:, 1]]).tolist())
train_set = np.random.permutation(train_set)
val_set = train_set[int(0.95*train_set.shape[0]):]
train_set = train_set[:int(0.95*train_set.shape[0])]

# assert set(train_set[:, 1]).issuperset(np.concatenate([test_set[:, 1], val_set[:, 1]]))
items = np.array(list(items))
train_dict = defaultdict(list)
test_dict = defaultdict(list)
test_que = defaultdict(list)
val_dict = defaultdict(list)

for ratings, dictionary in zip([train_set, test_set, test_que, val_set], [train_dict, test_dict, train_set_que, val_dict]):
    for user, item, rating in ratings:
        dictionary[user].append((item, rating))

train_users = {}
test_users = {}
val_users = {}

users = sorted(set(train_dict.keys()).union(test_dict.keys()).union(val_dict.keys()))

for user in users:
    if user in train_dict:
        train_users[user] = User(user, user, train_dict[user])
    if user in test_dict:
        ratings = test_que[user]
        if len(ratings) <= 0:
            ratings = [(item, 1) for item in np.random.choice(items, size=20, replace=False)]
        test_users[user] = LeaveOneOutUser(user, ratings, test_dict[user])

        assert user not in train_users, 'should not occur'

        train_users[user] = User(user, user, ratings)
    if user in val_dict:
        val_users[user] = LeaveOneOutUser(user, train_users[user].ratings, val_dict[user])

train_users = list(train_users.values())
test_users = list(test_users.values())
val_users = list(val_users.values())

# Create meta
# users = list(range(max(train_youusers)))
entity_range = list(range(max(items) + 1))
entities = [Entity(eid, eid, True, eid) for eid in entity_range]
meta = Meta(entities, train_users, [], {
	Sentiment.NEGATIVE: -1,
	Sentiment.POSITIVE: 1,
	Sentiment.UNSEEN: 0,
}, rated=entity_range)

feature = np.array([0 if user.index in test_dict else 1 for user in train_users])

name = 'beauty_warm'
save_pickle(os.path.join(name, 'meta.pickle'), meta)
save_pickle(os.path.join(name, 'fold_0', 'train.pickle'), train_users)
save_pickle(os.path.join(name, 'fold_0', 'test.pickle'), test_users)
save_pickle(os.path.join(name, 'fold_0', 'validation.pickle'), val_users)
np.save(os.path.join(name, 'mask.npy'), feature)