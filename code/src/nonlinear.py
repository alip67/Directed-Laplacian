# external files
import numpy as np
import pickle as pk
from scipy import sparse
import torch.optim as optim
from datetime import datetime
import os, time, argparse, csv
from collections import Counter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.datasets import WebKB, WikipediaNetwork, WikiCS
from tqdm import tqdm

# internal files
from utils.Citation import *
from layer.sparse_magnet import *
from utils.preprocess import geometric_dataset_sparse, load_syn
from utils.save_settings import write_log
from torch.autograd import Variable

from layer import model

# select cuda device if available
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="MagNet Conv (sparse version)")
    parser.add_argument('--log_root', type=str, default='../logs/', help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test', help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='code/dataset/data/tmp/', help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='WebKB/Cornell', help='data set selection')

    parser.add_argument('--epochs', type=int, default=3000, help='Number of (maximal) training epochs.')
    # parser.add_argument('--depth', type=int, default=3000, help='Number of (maximal) training epochs.')
    parser.add_argument('--q', type=float, default=0.25, help='q value for the phase matrix')
    parser.add_argument('--method_name', type=str, default='Magnet', help='method name')

    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='How many layers of gcn in the model, default 2 layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')

    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--fast', '-F', action='store_true', help='faster version of HyperGCN (True)')
    parser.add_argument('--mediators', action='store_true', help='True for Laplacian with mediators, False for Laplacian without mediators')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--l2', type=float, default=5e-4, help='l2 regularizer')

    parser.add_argument('--num_filter', type=int, default=32, help='num of filters')
    parser.add_argument('--randomseed', type=int, default=3407, help='if set random seed in training')  
    return parser.parse_args()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main(args):
    if args.randomseed > 0:
        torch.manual_seed(args.randomseed)
    
    date_time = datetime.now().strftime('%m-%d-%H:%M:%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if os.path.isdir(log_path) == False:
        try:
            os.makedirs(log_path)
        except FileExistsError:
            print('Folder exists!')

    load_func, subset = args.dataset.split('/')[0], args.dataset.split('/')[1]
    if load_func == 'WebKB':
        func = WebKB
    elif load_func == 'cora_ml':
        func = citation_datasets
    elif load_func == 'citeseer_npz':
        func = citation_datasets
    elif load_func == 'syn':
        func = load_syn
        args.data_path = args.data_path+'syn/'+subset
    else:
        print("wrong dataset name !!!")
        return

    dataset, X, label, train_mask, val_mask, test_mask, L = geometric_dataset_sparse(args.q, args.K, 
                            root=args.data_path+load_func, subset=subset,
                            dataset = func, load_only = False, save_pk = True)
    
    # if load_func == 'cora_ml':
    #     dataset = dataset[0]
    # elif load_func == 'citeseer_npz':
    #     dataset = dataset[0]
    # else:
    #     print("wrong dataset name !!!")
    #     return
  
    # normalize label, the minimum should be 0 as class index
    _label_ = label - np.amin(label)
    cluster_dim = np.amax(_label_)+1

    label = torch.from_numpy(_label_[np.newaxis]).to(device)
    criterion = nn.NLLLoss()

    splits = train_mask.shape[1]
    if len(test_mask.shape) == 1:
        #data.test_mask = test_mask.unsqueeze(1).repeat(1, splits)
        test_mask = np.repeat(test_mask[:,np.newaxis], splits, 1)

    results = np.zeros((splits, 4))
    for split in tqdm(range(splits)):
        log_str_full = ''


        # # initialise HyperGCN
        HyperGCN, X_norm = model.initialise(dataset,X, label,cluster_dim, args)
        # X_norm = torch.FloatTensor(X).to(device)
        X_norm = X_norm.to(device)
        



        # # train and test HyperGCN
        # HyperGCN = model.train(HyperGCN, dataset, train, args)
        # acc = model.test(HyperGCN, dataset, test, args)
        # print("accuracy:", float(acc), ", error:", float(100*(1-acc)))

        # model = MagNet(X_real.size(-1), L_real, L_img, K = args.K, label_dim=cluster_dim, layer = args.layer,
        #                 num_filter = args.num_filter, dropout=args.dropout).to(device)    

        modelnl = HyperGCN['model']
        opt = HyperGCN['optimiser']

        best_test_acc = 0.0
        train_index = train_mask[:,split]
        val_index = val_mask[:,split]
        test_index = test_mask[:,split]

        #################################
        # Train/Validation/Test
        #################################
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in tqdm(range(args.epochs)):
            start_time = time.time()
            ####################
            # Train
            ####################
            count, train_loss, train_acc = 0.0, 0.0, 0.0

            # for loop for batch loading
            count += np.sum(train_index)

            modelnl.train()
            preds, loss_encoder = modelnl(X_norm)
            # train_loss = F.nll_loss(preds[train_index,:], label[:,train_index].squeeze())+ loss_encoder
            train_loss = F.nll_loss(preds[train_index,:], label[:,train_index].squeeze())
            pred_label = preds.max(dim = 1)[1]
            train_acc = 1.0*((pred_label.unsqueeze(0)[:,train_index] == label[:,train_index])).sum().detach().item()/count
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
            outstrtrain = 'Train loss:, %.6f, acc:, %.3f,' % (train_loss.detach().item(), train_acc)
            #scheduler.step()
            ####################
            # Validation
            ####################
            modelnl.eval()
            count, test_loss, test_acc = 0.0, 0.0, 0.0
            
            # for loop for batch loading
            count += np.sum(val_index)
            preds,loss_encoder = modelnl(X_norm)
            pred_label = preds.max(dim = 1)[1]

            # test_loss = F.nll_loss(preds[val_index,:], label[:,val_index].squeeze()) + loss_encoder
            test_loss = F.nll_loss(preds[val_index,:], label[:,val_index].squeeze()) 
            test_acc = 1.0*((pred_label.unsqueeze(0)[:,val_index] == label[:,val_index])).sum().detach().item()/count

            outstrval = ' Test loss:, %.6f, acc:, %.3f,' % (test_loss.detach().item(), test_acc)
            
            duration = "---, %.4f, seconds ---" % (time.time() - start_time)
            log_str = ("%d ,/, %d ,epoch," % (epoch, args.epochs))+outstrtrain+outstrval+duration
            log_str_full += log_str + '\n'
            #print(log_str)

            ####################
            # Save weights
            ####################
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(modelnl.state_dict(), log_path + '/model'+str(split)+'.t7')
            else:
                early_stopping += 1
            if early_stopping > 500 or epoch == (args.epochs-1):
                torch.save(modelnl.state_dict(), log_path + '/model_latest'+str(split)+'.t7')
                break

        write_log(vars(args), log_path)

        ####################
        # Testing
        ####################
        modelnl.load_state_dict(torch.load(log_path + '/model'+str(split)+'.t7'))
        modelnl.eval()
        preds, loss_encoder = modelnl(X_norm)
        pred_label = preds.max(dim = 1)[1]
        np.save(log_path + '/pred' + str(split), pred_label.to('cpu'))
    
        count = np.sum(val_index)
        acc_train = (1.0*((pred_label.unsqueeze(0)[:,val_index] == label[:,val_index])).sum().detach().item())/count

        count = np.sum(test_index)
        acc_test = (1.0*((pred_label.unsqueeze(0)[:,test_index] == label[:,test_index])).sum().detach().item())/count

        modelnl.load_state_dict(torch.load(log_path + '/model_latest'+str(split)+'.t7'))
        modelnl.eval()
        preds, loss_encoder = modelnl(X_norm)
        pred_label = preds.max(dim = 1)[1]
        np.save(log_path + '/pred_latest' + str(split), pred_label.to('cpu'))
    
        count = np.sum(val_index)
        acc_train_latest = (1.0*((pred_label.unsqueeze(0)[:,val_index] == label[:,val_index])).sum().detach().item())/count

        count = np.sum(test_index)
        acc_test_latest = (1.0*((pred_label.unsqueeze(0)[:,test_index] == label[:,test_index])).sum().detach().item())/count

        ####################
        # Save testing results
        ####################
        logstr = 'val_acc: '+str(np.round(acc_train, 3))+' test_acc: '+str(np.round(acc_test,3))+' val_acc_latest: '+str(np.round(acc_train_latest,3))+' test_acc_latest: '+str(np.round(acc_test_latest,3))
        print(logstr)
        results[split] = [acc_train, acc_test, acc_train_latest, acc_test_latest]
        log_str_full += logstr
        with open(log_path + '/log'+str(split)+'.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1

    dir_name = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../result_arrays',args.log_path,args.dataset+'/')
    args.log_path = os.path.join(args.log_path,args.method_name, args.dataset)

    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')
    save_name = args.method_name + 'lr' + str(int(args.lr*1000)) + 'num_filters' + str(int(args.num_filter)) + 'q' + str(int(100*args.q)) + 'layer' + str(int(args.layer)) + 'K' +  str(int(args.K))
    args.save_name = save_name
    results = main(args)
    np.save(dir_name+save_name, results)