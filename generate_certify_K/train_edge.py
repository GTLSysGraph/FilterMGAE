from __future__ import division
from __future__ import print_function

# 只能看到generate_certify_k里面的内容，所以添加最顶层的目录,这样该文件夹和上层都可见，该文件夹当中直接import即可
import sys
sys.path.append("..")

from datasets_dgl.data_dgl import load_attack_data
from easydict              import EasyDict
import dgl
import time
import argparse
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import os

from models import GCN
from core_Ber import Smooth_Ber
from utils import *



parser = argparse.ArgumentParser()
#parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--outdir', type=str, default='./dir_model', help='folder to save model and training log)')

parser.add_argument('--cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument("--batch", type=int, default=10000, help="batch size")
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--prob', default=0.8, type=float,
                    help="probability to keep the status for each binary entry")
parser.add_argument('--beta', default=0.0, type=float,
                    help="propagation factor")

parser.add_argument("--predictfile", type=str, help="output prediction file")
parser.add_argument("--certifyfile", type=str, help="output certified file")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=20)
parser.add_argument("--N", type=int, default=20, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.01, help="failure probability")

args = parser.parse_args()
print(args.cuda)




np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    torch.cuda.manual_seed(args.seed)


################# no attack
dataset_name = 'Attack-Cora'
no_attack    = 'Meta_Self-0.0'

DATASET0 = EasyDict()
DATASET0.ATTACK = {
    "data":dataset_name,
    "attack":no_attack.split('-')[0],
    "ptb_rate":no_attack.split('-')[1]
}
dataset  = load_attack_data(DATASET0['ATTACK'])
graph = dataset.graph

# 先去掉自环
graph = dgl.remove_self_loop(graph)
degrees = graph.in_degrees()

adj = to_scipy(graph.adj().to_dense())
adj = sparse_mx_to_torch_sparse_tensor(adj)
adj_norm = degree_normalize_sparse_tensor(adj) # train GCN的时候用norm的


features   = graph.ndata['feat']
labels     = graph.ndata['label']
idx_train  = graph.ndata['train_mask'].nonzero().squeeze()
idx_val    = graph.ndata['val_mask'].nonzero().squeeze()

sorted_value, sorted_indices = torch.sort(graph.in_degrees())
print(sorted_value)
print(sorted_indices)
idx_test   = sorted_indices[-20:]

# idx_test   = graph.ndata['test_mask'].nonzero().squeeze()[:40]
# idx_test = torch.arange(graph.num_nodes())


if args.cuda:
    #model.cuda()
    features = features.cuda()
    adj_norm = adj_norm.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
else:
    pass


# GCN
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()


def get_num_classes_dim(dataset_name):
    """Return the number of classes in the dataset. """
    if dataset_name == "Attack-Cora":
        num_class, dim = 7, 1433
    if dataset_name == "Attack-Citeseer":
        num_class, dim = 6, 3703
    elif dataset_name == "Attack-Pubmed":
        num_class, dim = 3, 500

    return num_class, dim



def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj_norm)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()


    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj_norm)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    

def test():
    model.eval()
    output = model(features, adj_norm)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


def certify():
    num_class, dim = get_num_classes_dim(dataset_name)
    # create the smoothed classifier g
    smoothed_classifier = Smooth_Ber(model, num_class, dim, args.prob, adj, features, args.cuda)

    # certifyfile = './dir_certify/certifyfile_' + dataset_name  +  '_' + str(args.prob) + '_N0_' + str(args.N0) + '_N_' + str(args.N)

    certifyfile = './dir_certify/test'

    # prepare output file
    f = open(certifyfile, 'w')
    print("idx\tlabel\tpredict\tpABar\tcorrect\tdegree\ttime", file=f, flush=True)

    cnt = 0
    cnt_certify = 0
    
    for i in idx_test:
    #for i in idx_test[:10]:
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        before_time = time.time()
        # make the prediction
        prediction, pABar = smoothed_classifier.certify_Ber(i, args.N0, args.N, args.alpha, args.batch)
        #print(prediction, labels[i])
        after_time = time.time()
        correct = int(prediction == labels[i])

        cnt += 1
        cnt_certify += correct

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i, labels[i], prediction, pABar, correct, degrees[i], time_elapsed), file=f, flush=True)

    f.close()

    print("certify acc:", float(cnt_certify) / cnt)



if __name__ == "__main__":

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):

        train(epoch)
        
        torch.save({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.prob.'+str(args.prob)))


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    #### Testing
    test()


    ## Certify
    certify()