import json
import pickle

import numpy as np

models = ['ginrec-inner']
# models = ['ginrec-a0', 'ginrec-a001', 'ginrec-a01', 'ginrec-a05', 'ginrec-a1', 'ginrec-a2']
		  #['ginrec-no-g', 'ginrec-no-r-g', 'ginrec-bipartite']
# path_to_file = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}' \
# 			   '_features_key_user_state_ab_warm_start_parameter_ml_mr_1m_warm_start_model_idcf' \
# 			   '{1}_metrics.pickle'
# path_to_file = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}' \
# 			   '_features_key_user_state_ml_mr_1m_warm_start_parameter_ml_mr_1m_warm_start_model_idcf' \
# 			   '{1}_metrics.pickle'
# ml ablation
path_to_file = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}' \
			   '_features_complex_features_graphsage_item_state_ab_warm_start_parameter_ml_mr_1m_warm_start_model_simplerec' \
			   '{1}_metrics.pickle'

# path_to_file = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}{1}_metrics.pickle'
# ab ablation
# path_to_file = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}' \
# 			   '_features_key_user_state_ab_warm_start_parameter_ml_mr_1m_warm_start' \
# 			   '{1}_model_idcf_metrics.pickle'

# path_to_file = '../results/{0}/{{0}}/fold_0_{{0}}_features_processed_state_ml_mr_1m_warm_start_parameter_ml_mr_1m_warm_start_model_simplerec_metrics.pickle'
datasets = ['ab_user_cold_start']#, 'ml_user_cold_start']'
metrics = ['ndcg', 'recall', 'precision', 'cov']
study = ''
ats = [20]

for dataset in datasets:
	print(f'--------{dataset}--------')
	path_to_dataset = path_to_file.format(dataset, study)
	for model in models:
		print(f'----{model}----')
		with open(path_to_dataset.format(model), 'rb') as fp:
			data = pickle.load(fp)
			res = {}
			for metric in metrics:
				if study.__contains__('idcf_neg_sampling') and metric in ['cov', 'auc']:
					continue
				res[metric] = {}
				if not study.__contains__('idcf_neg_sampling'):
					for at in ats:
						if metric == 'auc':
							r = np.average(data[metric])
						elif metric == 'cov':
							r = data[metric][at]
						else:
							r = np.average([r[at] for r in data[metric]])
						res[metric][at] = '{0:.5f}'.format(np.round(r, decimals=5))
				else:
					res[metric] = '{0:.5f}'.format(np.round(np.average([r[max(r)] for r, n in zip(data[metric], data['n_pos'])]), decimals=5))
					# res[metric] = np.round(np.average([r[20] for r in data[metric]]), decimals=5)

		# print(json.dumps(res,indent=2))
		print(res)