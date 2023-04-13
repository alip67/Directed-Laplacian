from nl import networks
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F

from torch.autograd import Variable
from tqdm import tqdm
from nl import utilsnl



def train(HyperGCN, dataset, T, args):
    """
    train for a certain number of epochs

    arguments:
	HyperGCN: a dictionary containing model details (gcn, optimiser)
	dataset: the entire dataset
	T: training indices
	args: arguments

	returns:
	the trained model
    """    
    
    hypergcn, optimiser = HyperGCN['model'], HyperGCN['optimiser']
    hypergcn.train()
    
    X, Y = dataset['features'], dataset['labels']

    for epoch in tqdm(range(args.epochs)):

        optimiser.zero_grad()
        Z = hypergcn(X)
        loss = F.nll_loss(Z[T], Y[T])

        loss.backward()
        optimiser.step()

    HyperGCN['model'] = hypergcn
    return HyperGCN



def test(HyperGCN, dataset, t, args):
    """
    test HyperGCN
    
    arguments:
	HyperGCN: a dictionary containing model details (gcn)
	dataset: the entire dataset
	t: test indices
	args: arguments

	returns:
	accuracy of predictions    
    """
    
    hypergcn = HyperGCN['model']
    hypergcn.eval()
    X, Y = dataset['features'], dataset['labels']
    

    Z = hypergcn(X) 
    return accuracy(Z[t], Y[t])
    


def accuracy(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels

    returns: 
    accuracy
    """
    
    predictions = Z.max(1)[1].type_as(Y)
    correct = predictions.eq(Y).double()
    correct = correct.sum()

    accuracy = correct / len(Y)
    return accuracy



def initialise(dataset,X, label,train_idx,cluster_dim, args):
    """
    initialises GCN, optimiser, normalises graph, and features, and sets GPU number
    
    arguments:
    dataset: the entire dataset (with graph, features, labels as keys)
    args: arguments

    returns:
    a dictionary with model details (hypergcn, optimiser)    
    """
    
    HyperGCN = {}
    V, E =dataset[0].y.size(-1), dataset[0].edge_index
    X, Y = X, label

    # hypergcn and optimiser
    args.d, args.c = X.shape[1], cluster_dim
    hypergcn = networks.HyperGCN(V, E, X, args)
    optimiser = optim.Adam(list(hypergcn.parameters()), lr=args.lr, weight_decay=args.decay)


    # node features in sparse representation
    X_norm = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X_norm = torch.FloatTensor(np.array(X_norm.todense()))
    
    # # labels
    # Y = np.array(Y.cpu())
    # Y = torch.LongTensor(np.where(Y)[1])

    # cuda
    args.Cuda = torch.cuda.is_available()
    if args.Cuda:
        hypergcn.cuda()
        # X, Y = X.cuda(), Y.cuda()

    # # update dataset with torch autograd variable
    # dataset.data.features = Variable(X_norm)
    # dataset.data.labels = Variable(Y)

    # update model and optimiser
    HyperGCN['model'] = hypergcn
    HyperGCN['optimiser'] = optimiser
    return HyperGCN, X_norm



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)
