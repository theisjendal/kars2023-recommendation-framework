import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

model1 = 'idcf-i'
model2 = 'inreg-act'
path_to_m1 = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}_features_key_user_state_ml_mr_1m_warm_start_parameter_ml_mr_1m_warm_start_model_idcf{1}_metrics.pickle'
path_to_m2 = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}_features_processed_state_ml_mr_1m_warm_start_parameter_ml_mr_1m_warm_start{1}_metrics.pickle'
# path_to_m1 = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}_features_processed_state_ml_mr_1m_warm_start_parameter_ml_mr_1m_warm_start{1}_model_simplerec_metrics.pickle'
# path_to_m2 = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}_features_processed_state_ml_mr_1m_warm_start_parameter_ml_mr_1m_warm_start{1}_model_simplerec_metrics.pickle'
# path_to_m1 = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}_features_graphsage_state_ab_warm_start_parameter_ml_mr_1m_warm_start{1}_metrics.pickle'
# path_to_m2 = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}_features_complex_features_graphsage_item_state_ab_warm_start_parameter_ml_mr_1m_warm_start{1}_model_simplerec_metrics.pickle'
# path_to_m2 = '../../../upload_folder/{0}/{{0}}/fold_0_{{0}}_features_complex_features_graphsage_item_state_ab_warm_start_parameter_ml_mr_1m_warm_start{1}_metrics.pickle'
datasets = ['ml_user_cold_start']#, 'ml_user_cold_start']
metrics = ['ndcg', 'recall', 'precision']
# metrics = ['ndcg']
# study = '_idcf_neg_sampling'
study = ''
ats = [20]

np.random.seed(42)

p = 0.05

for dataset in datasets:
	print(f'--------{dataset}--------')

	with open(path_to_m1.format(dataset, study).format(model1), 'rb') as fp1,\
			open(path_to_m2.format(dataset, study).format(model2), 'rb') as fp2:
		data1 = pickle.load(fp1)
		data2 = pickle.load(fp2)
		for metric in metrics:
			for at in ats:
				if not study.endswith('idcf_neg_sampling'):
					d1 = np.array([[u, r[at][0]] for u, r in sorted(zip(data1['user'], data1[metric]))])
					d2 = np.array([[u, r[at][0]] for u, r in sorted(zip(data2['user'], data2[metric]))])
				else:
					if metric in ['cov', 'auc']:
						continue

					d1 = np.array([[u, r[max(r)][0]] for u, r in sorted(zip(data1['user'], data1[metric]))])
					d2 = np.array([[u, r[max(r)][0]] for u, r in sorted(zip(data2['user'], data2[metric]))])

				assert np.all(d1[:, 0] == d2[:, 0])
				d1, d2 = d1[:,1], d2[:,1]
				# a, b = np.histogram(d2)
				# plt.hist(d2)
				# plt.show()
				print(f'{metric}@{at}: ')
				# H_0 d1 > d2, alternative (H_1) d1 < d2. Output p value for H_0
				# print(stats.mannwhitneyu(d1, d2, alternative='less'))
				# s1 = np.random.choice(d1, 10*len(d1))
				# s2 = np.random.choice(d2, 10*len(d2))
				# print(stats.mannwhitneyu(s1, s2, alternative='less'))
				ttest1 = stats.wilcoxon(d1, d2, alternative='less')
				is_better = np.less_equal(ttest1.pvalue, p)

				if is_better:
					print(f'{model2} is statistically better than model 1 ({ttest1.pvalue})')
				else:
					ttest2 = stats.wilcoxon(d2, d1, alternative='less')
					is_better = np.less_equal(ttest2.pvalue, p)

					if is_better:
						print(f'{model1} is statistically better than model 2 ({ttest2.pvalue})')
					else:
						print(f'Models are more or less equal. ({ttest1.pvalue}) ({ttest2.pvalue})')
				# print(stats.wilcoxon(d2, d1, alternative='less'))
				# print(sum(d2 >= d1) / len(d1))

