import os
import time
import numpy as np
import random
import argparse
import torch
from model.trainer import Trainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker, create_cross_Graph


method_name = 'CoPD' 
parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--dataset', type=str, default='phone_electronic, sport_phone, sport_cloth, electronic_cloth', help='')

# model part
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--GNN', type=int, default=3, help='GNN layer.')
parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
# train part
parser.add_argument('--num_epoch', type=int, default=50, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--lambda1', type=float, default=1.0, help='')
parser.add_argument('--lambda2', type=float, default=1.0, help='')
parser.add_argument('--reg_weight', type=float, default=1e-3, help="the weight decay of BPR Loss")
parser.add_argument('--temp', type=float, default=0.05, help='')
# save part
parser.add_argument('--log', type=str, default='logs/{}'.format(method_name), help='log directory')

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)

args = parser.parse_args()
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])

# made log
log = os.path.join(args.log, '{}_{}_{}_{}_{}'.format(args.dataset, args.GNN, args.lambda1, args.lambda2, args.temp))
if os.path.isdir(log):
    print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
    time.sleep(5)
    os.system('rm -rf %s/' % log)

os.makedirs(log)
print("made the log directory", log)
print(opt)
with open(log + '/tmp.txt', 'a') as f:
    f.write(str(opt))

datasetname  = opt["dataset"]
source_G = GraphMaker(opt, datasetname)
source_UV = source_G.UV
source_VU = source_G.VU
source_adj = source_G.adj

datasetname = datasetname.split("_")
datasetname = datasetname[1] + "_" + datasetname[0]
target_G = GraphMaker(opt, datasetname)
target_UV = target_G.UV
target_VU = target_G.VU
target_adj = target_G.adj

cross_adj = create_cross_Graph(source_UV, source_VU, target_UV, target_VU, opt['dataset'])

print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'], opt['batch_size'], opt, evaluation = -1)
source_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 1)
target_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 2)

print("user_num", opt["source_user_num"])
print("source_item_num", opt["source_item_num"])
print("target_item_num", opt["target_item_num"])
print("source train data : {}, target train data {}, source test data : {}, source test data : {}".format(len(train_batch.source_train_data),len(train_batch.target_train_data),len(train_batch.source_test_data),len(train_batch.target_test_data)))

source_adj = source_adj.cuda()
target_adj = target_adj.cuda()
cross_adj = cross_adj.cuda()

# model
trainer = Trainer(opt, source_adj, target_adj, cross_adj)

max_steps = len(train_batch) * opt['num_epoch']

best_s_score = [0, 0]
best_t_score = [0, 0]
best_epoch_s = 0
best_epoch_t = 0
# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    start_time = time.time()
    batch_all_loss = []
    for i, batch in enumerate(train_batch):
        loss = trainer.reconstruct_graph(batch)
        batch_all_loss.append(loss)

    epoch_all_loss = np.mean(batch_all_loss)
    duration = time.time() - start_time
    print('epoch:{}, time:{:.2f}, loss:{:.4f}'.format(epoch, duration, epoch_all_loss))
    with open(log + '/tmp.txt', 'a') as f:
        f.write('epoch:{}, time:{:.2f}, loss:{:.4f}\n'.format(epoch, duration, epoch_all_loss))

    if epoch <= opt['num_epoch'] - 10: # test from final 10 epochs 
    # if epoch % 1:
        # pass
        continue

    # eval model
    trainer.model.eval()
    trainer.evaluate_embedding()

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    for i, batch in enumerate(source_dev_batch):
        predictions = trainer.source_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()
            valid_entity += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    s_ndcg = NDCG / valid_entity
    s_hit = HT / valid_entity
    if s_ndcg > best_s_score[1]:
        best_s_score[1] = s_ndcg
        best_s_score[0] = s_hit
        best_epoch_s = epoch
        torch.save(trainer.model.state_dict(), os.path.join(log, 'best_ndcg1.pkl'))

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    for i, batch in enumerate(target_dev_batch):
        predictions = trainer.target_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()
            valid_entity += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    t_ndcg = NDCG / valid_entity
    t_hit = HT / valid_entity
    if t_ndcg > best_t_score[1]:
        best_t_score[1] = t_ndcg
        best_t_score[0] = t_hit
        best_epoch_t = epoch
        torch.save(trainer.model.state_dict(), os.path.join(log, 'best_ndcg2.pkl'))
    
    print('test: hr1:{:.4f},ndcg1:{:.4f}, hr2:{:.4f},ndcg2:{:.4f}'.format(s_hit, s_ndcg, t_hit, t_ndcg))
    with open(log + '/tmp.txt', 'a') as f:
        f.write('test: hr1:{:.4f},ndcg1:{:.4f}, hr2:{:.4f},ndcg2:{:.4f}\n'.format(s_hit, s_ndcg, t_hit, t_ndcg))

print('best epoch1 {}: hr1:{:.4f},ndcg1:{:.4f}, best epoch2 {}: hr2:{:.4f},ndcg2:{:.4f}'.\
      format(best_epoch_s, best_s_score[0], best_s_score[1], best_epoch_t, best_t_score[0], best_t_score[1]))
with open(log + '/tmp.txt', 'a') as f:
        f.write('best epoch1 {}: hr1:{:.4f},ndcg1:{:.4f}, best epoch2 {}: hr2:{:.4f},ndcg2:{:.4f}'.\
            format(best_epoch_s, best_s_score[0], best_s_score[1], best_epoch_t, best_t_score[0], best_t_score[1]))
print(opt)

