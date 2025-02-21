import numpy as np
#from tqdm import trange
import argparse
from decimal import Decimal

# parser = argparse.ArgumentParser(description='Get thresholds')
# parser.add_argument('--r_start', default=0, type=int, 
#                     help='[r_start, r_end)')
# parser.add_argument('--r_end', default=21, type=int, 
#                     help='[r_start, r_end)')
# parser.add_argument('--a', type=int, default=80,
#                     help='alpha = a / 100')
# args = parser.parse_args()

# v_range = [args.r_start, args.r_end]

# fn = 'cora'
# global_d =  2708 # cora

# # fn = 'citeseer'
# # global_d = 3327 # citeseer

# # fn = 'pubmed'
# # global_d = 19717 # pubmed


def certify_K(p_l, frac_alpha, global_d, v_range, fn):

	frac_beta = (1 - frac_alpha)
	
	alpha = int(frac_alpha * 100)
	beta = 100 - alpha

	for v in range(v_range[0], v_range[1]):

		plower_Z = int(p_l * 100 ** 10) * (100 ** (global_d-10))
		pupper_Z = int((1-p_l) * 100 ** 10) * (100 ** (global_d-10))
		total_Z = 100 ** global_d

		complete_cnt = []
		cnt = np.load('./dir_list_counts/{}/complete_count_{}.npy'.format(fn, v), allow_pickle=True)
		complete_cnt += list(cnt)
		
		raw_cnt = 0
		
		outcome = []
		for ((s, t), c) in complete_cnt:
			outcome.append((
				# likelihood ratio x flips s, x bar flips t
				# and then count, s, t
				(alpha ** (t - s)) * (beta ** (s - t)), c, s, t
			))
			if s != t:	
				outcome.append((
					(alpha ** (s - t)) * (beta ** (t - s)), c, t, s
				))

			raw_cnt += c
			if s != t:
				raw_cnt += c

		# sort likelihood ratio in a descending order, i.e., r1 >= r2 >= ...

		outcome_descend = sorted(outcome, key = lambda x: -x[0])

		p_given_lower = 0
		q_given_lower = 0
		for i in range(len(outcome_descend)):
			ratio, cnt, s, t = outcome_descend[i]
			p = (alpha ** (global_d - s)) * (beta ** s)
			q = (alpha ** (global_d - t)) * (beta ** t)
			q_delta_lower = q * cnt
			p_delta_lower = p * cnt

			if p_given_lower + p_delta_lower < plower_Z:
				p_given_lower += p_delta_lower
				q_given_lower += q_delta_lower
			else:
				q_given_lower += (plower_Z - p_given_lower) / Decimal(ratio)
				#q_given_lower += q * (plower_Z - p_given_lower) / Decimal(p)
				break
		q_given_lower /= total_Z


		# sort likelihood ratio in a ascending order
		outcome_ascend = sorted(outcome, key = lambda x: x[0])
		p_given_upper = 0
		q_given_upper = 0
		for i in range(len(outcome_ascend)):
			ratio, cnt, s, t = outcome_ascend[i]
			p = (alpha ** (global_d - s)) * (beta ** s)
			q = (alpha ** (global_d - t)) * (beta ** t)
			q_delta_upper = q * cnt
			p_delta_upper = p * cnt

			if p_given_upper + p_delta_upper < pupper_Z:
				p_given_upper += p_delta_upper
				q_given_upper += q_delta_upper
			else:
				q_given_upper += (pupper_Z - p_given_upper) / Decimal(ratio)
				#q_given_upper += q * (pupper_Z - p_given_upper) / Decimal(p)
				break
		q_given_upper /= total_Z

		print(q_given_lower, q_given_upper)

		if q_given_lower - q_given_upper < 0:
			return v


# v_range = [0, 21]
# fn = 'cora'
# global_d =  2708 # cora

# # fn = 'citeseer'
# # global_d = 3327 # citeseer

# # fn = 'pubmed'
# # global_d = 19717 # pubmed

# p_l = 0.9
# #p_l = 0.7857142857142857142857142857
# alpha = 0.7
# v = certify_K(p_l, alpha, global_d, v_range, fn)
# print('Certified K:', v)