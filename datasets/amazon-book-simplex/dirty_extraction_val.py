import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.entity import Entity
from shared.enums import Sentiment
from shared.meta import Meta
from shared.user import User, LeaveOneOutUser
from shared.utility import save_pickle

user_df = pd.read_csv('user_list.txt', sep=' ')
idx_org = user_df.set_index('remap_id')['org_id'].to_dict()

train_ratings = []
test_ratings = []

train_users = {}
test_users = {}

#load train and test
for f_name in ['train.txt', 'test.txt']:
	with open(f_name) as f:
		for line in tqdm(f, desc=f_name):
			line = list(filter(None, line.strip('\n').split(' ')))
			if len(line) <= 1:
				continue

			u_idx = int(line[0])
			ratings = [(int(item), 1) for item in line[1:]]
			if f_name.startswith('train'):
				train_users[u_idx] = User(idx_org[u_idx], u_idx, ratings)
			else:
				test_users[u_idx] = LeaveOneOutUser(u_idx, train_users[u_idx].ratings, ratings)

ratings = np.array([(user.index, item, rating) for user in train_users.values() for item, rating in user.ratings])
found_split = False
seed = 40
progress = tqdm(desc='Creating validation split')
val_users = {}
np.random.seed(seed)
random.seed(seed)
permutation = np.random.permutation(len(ratings))
split = int(len(permutation) *0.95)
train_indices = permutation[:split]
val_indices = permutation[split:]
while not found_split:
	progress.update()
	train_indices = np.sort(train_indices)
	val_indices = np.sort(val_indices)

	train_ratings = ratings[train_indices]
	val_ratings = ratings[val_indices]

	difference = set(np.unique(val_ratings[:, 1])).difference(np.unique(train_ratings[:, 1]))
	found_split = len(difference) == 0

	if found_split:
		continue

	# Swap one val rating of item without any train ratings to train set and move one back
	new_vals = np.random.choice(np.arange(len(train_indices)), len(difference), replace=False)
	for idx, item in enumerate(difference):
		args = np.argwhere(val_ratings[:, 1] == item)[0]  # same order as indices
		v_arg = np.random.choice(args, 1)
		t_arg = new_vals[idx]
		v_idx = val_indices[v_arg]
		val_indices[v_arg] = train_indices[t_arg]
		train_indices[t_arg] = v_idx

	assert set(train_indices).isdisjoint(val_indices)
progress.close()

# Get unique users and count for train and validation
utu, utc = np.unique(train_ratings[:, 0], return_counts=True, )
uvu, uvc = np.unique(val_ratings[:, 0], return_counts=True)
assert len(utu) == len(train_users), 'Expected all users to be in train set after creating validation set but ' \
									 'some are missing'

t_idx = 0  # Current place in train ratings
v_idx = 0  # Current place in val ratings
vu_idx = 0  # Current place in unique validation users
for u_idx, count in tqdm(zip(utu, utc), 'Validation'):
	user = train_users[u_idx]
	ratings = [(item, rating) for _, item, rating in train_ratings[t_idx:t_idx+count]]
	user.ratings = ratings
	t_idx += count

	if len(uvu) <= vu_idx:
		continue

	val_user = uvu[vu_idx]

	if u_idx == val_user:
		v_count = uvc[vu_idx]
		v_ratings = [(item, rating) for _, item, rating in val_ratings[v_idx:v_idx+v_count]]
		val_users[u_idx] = LeaveOneOutUser(u_idx, ratings, v_ratings)
		v_idx += v_count
		vu_idx += 1

train_users = list(train_users.values())
val_users = list(val_users.values())
test_users = list(test_users.values())

# Create meta
# users = list(range(max(train_youusers)))
entities = list(range(max([item for user in train_users for item, _ in user.ratings]) + 1))
entities = [Entity(eid, eid, True, eid) for eid in entities]
meta = Meta(entities, train_users, [], {
	Sentiment.NEGATIVE: -1,
	Sentiment.POSITIVE: 1,
	Sentiment.UNSEEN: 0,
})

# save
name = 'val_warm'
save_pickle(os.path.join(name, 'meta.pickle'), meta)
save_pickle(os.path.join(name, 'fold_0', 'train.pickle'), train_users)
save_pickle(os.path.join(name, 'fold_0', 'test.pickle'), test_users)
save_pickle(os.path.join(name, 'fold_0', 'validation.pickle'), val_users)