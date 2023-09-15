import os

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

train_users = list(train_users.values())
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
name = 'test_warm'
save_pickle(os.path.join(name, 'meta.pickle'), meta)
save_pickle(os.path.join(name, 'fold_0', 'train.pickle'), train_users)
save_pickle(os.path.join(name, 'fold_0', 'test.pickle'), test_users)
save_pickle(os.path.join(name, 'fold_0', 'validation.pickle'), test_users)