import pandas as pd
import sys
import numpy as np

from certify_K import certify_K

v_range = [0, 21]

# fn = 'cora'
# global_d =  2485 # cora

fn = 'citeseer'
global_d = 2110 # citeseer

# fn = 'pubmed'
# global_d = 19717 # pubmed


file_path = './dir_certify/certifyfile_Attack-Citeseer_0.999_N0_20_N_100'
# file_path = './dir_certify/certifyfile_Attack-Cora_0.999_N0_20_N_200'
# file_path = './dir_certify/test'


df = pd.read_csv(file_path, sep="\t")

accurate = df["correct"]
predict = df["predict"]
label = df["label"]
pAHat = df["pABar"]
#
degrees = df["degree"]


#print(pAHat)

test_num = predict == label
#print(sum(test_num))
test_acc = sum(test_num) / float(len(label))
#print(sum(df["correct"]))
print('certify acc:', sum(accurate))

alpha = float(sys.argv[1])
print('alpha = {}'.format(alpha))

K = np.zeros(v_range[1], dtype=int)


file_getK_path = './dir_get_K/' + fn
f = open(file_getK_path, 'w')
print("idx\tCertifiedK\tDegree", file=f, flush=True)



for idx in range(len(pAHat)):
	if accurate[idx]:
		v = certify_K(pAHat[idx], alpha, global_d, v_range, fn)
		print('pAHat:', pAHat[idx], 'Certified K:', v)
		print("{}\t{}\t{}".format(idx, v, degrees[idx]), file=f, flush=True)
		K[v] += 1

f.close()


print(K)


K_cer = np.cumsum(K[::-1])[::-1]

print('###########################')
for idx in range(len(K_cer)):
	print(idx+1, K_cer[idx])