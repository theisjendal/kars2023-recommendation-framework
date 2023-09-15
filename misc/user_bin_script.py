import subprocess

base_str = 'source ../.venv/bin/activate; python user_bin_plot.py --dataset ../datasets --results ../../../upload_folder --experiment {0} --models ginrec ginrec-bipartite ginrec-bipartite-t pinsage idcf-i graphsage toppop --metric ndcg --at 20 --study {1}'

experiments = ['ml_user_cold_start_1250', 'ab_user_cold_start', 'ml_user_cold_start']
studies = ['user_group', 'user_sparsity', 'user_popularity']

for experiment in experiments:
	print(f'Experiment: {experiment}')
	for study in studies:
		print(f'Study: {study}')
		str_arg = base_str.format(experiment, study)

		if experiment == 'ml_user_cold_start' and study == 'user_popularity':
			str_arg += ' --legend'

		p = subprocess.Popen(str_arg, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
		# for line in p.stdout:
		# 	print(line)

		p.wait()