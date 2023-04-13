import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
import numpy as np
from numpy import linalg as LA
from scipy.sparse import coo_matrix, csr_matrix

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import networkx as nx
import random
from tqdm import tqdm
from multiprocessing import Pool
from typing import List

def get_net(d_in, lst_hidden_dim):
    layers = []
    layers = [ nn.Linear(d_in, lst_hidden_dim[0]), nn.ReLU()]
    for i in range(len(lst_hidden_dim) - 1): 
        layers.append(nn.Linear(lst_hidden_dim[i], lst_hidden_dim[i+1]))
        layers.append(nn.ReLU())
    # layers.append(nn.Linear(64, 8))
    return layers

def get_net_decoder(d_in, lst_hidden_dim):
    lst_hidden_dim.reverse()
    layers = []
    # layers = [ nn.Linear(lst_hidden_dim[0], lst_hidden_dim[1]), nn.ReLU()]
    for i in range(len(lst_hidden_dim) - 1): 
        layers.append(nn.Linear(lst_hidden_dim[i], lst_hidden_dim[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(lst_hidden_dim[-1], d_in))
    return layers

class autoencoder(nn.Module):
    def __init__(self,d_in,d_hidden, num_feat):
        super(autoencoder, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        lst_hidden_dim = [d_hidden]
        a = d_hidden

        while a>1:
            a = a//2
            lst_hidden_dim.append(a)

        #encoder
        self.encoder = nn.Sequential(*get_net(d_in,lst_hidden_dim))

        #decoder
        self.decoder = nn.Sequential(*get_net_decoder(d_in,lst_hidden_dim))

        # #encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(num_feat, 128),  # reduces from n * 724 to 128
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 1),
        # )

        # #decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(1, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_feat),
        #     # nn.Sigmoid()  # cause tensors are 0, 1
        # )

    def forward(self, X):
        X_enc_out = self.encoder(X)
        X = self.decoder(X_enc_out)

        return X, X_enc_out
    
class labelencoder(nn.Module):
    def __init__(self,d_in,d_hidden, num_feat):
        super(labelencoder, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden

        lst_hidden_dim = [d_hidden]
        a = d_hidden

        while a>1:
            a = a//2
            lst_hidden_dim.append(a)

        # #encoder
        # self.encoder = nn.Sequential(*get_net(d_in,lst_hidden_dim))

        # #decoder
        # self.decoder = nn.Sequential(*get_net_decoder(d_in,lst_hidden_dim))


        #encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_feat, 128),  # reduces from n * 724 to 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        #decoder
        self.decoder = nn.Sequential(
            nn.Linear(1, 5),
        )

        # #encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(d_in, d_hidden),  # reduces from n * 724 to 128
        #     nn.ReLU(),
        #     nn.Linear(d_hidden, d_hidden),
        #     nn.ReLU(),
        #     nn.Linear(d_hidden, d_hidden),
        #     nn.ReLU(),
        #     nn.Linear(d_hidden, 1)
        # )

        # #decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(1, d_hidden),
        #     nn.ReLU(),
        #     nn.Linear(d_hidden, d_hidden),
        #     nn.ReLU(),
        #     nn.Linear(d_hidden, d_hidden),
        #     nn.ReLU(),
        #     nn.Linear(d_hidden, d_in),
        #     # nn.Sigmoid()  # cause tensors are 0, 1
        # )

    def forward(self, X):
        X_enc_out = self.encoder(X)
        X = self.decoder(X_enc_out)



        return F.log_softmax(X, dim=1), X_enc_out


class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, feat_dim,reapproximate=True, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda
        # self.autoencoder = autoencoder(self.b, self.b//2,self.a)
        self.labelencoder = labelencoder(self.b, self.b//2,self.a)

        self.emb = nn.Linear(b, 1, bias=True)
        self.activation = nn.Tanh()

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        # self.lin = nn.Linear(a,b)
        self.reset_parameters()
        


    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.lin.weight)
        # nn.init.xavier_uniform_(self.lin.bias)
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        self.emb.reset_parameters()



    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)
        # HW = self.lin(H)
        embed = self.emb(HW)
        # embed = self.activation(embed)
        # embedding = self.embedding(HW)
        

        if self.reapproximate:
            n = H.shape[0]
            # n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian_Nonlinear_torch(n, structure, embed, m)
        else: A = structure

        print("grad for blockbox : " , self.emb.weight)

        if self.cuda: A = A.cuda()
        A = Variable(A)
        # print(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b
    
    def forward_autoencoder(self, structure, H, m=True):


        W, b = self.W, self.bias
        # HW = torch.mm(H, W)
        reconstruct_out, hidden_1dim_out = self.autoencoder(H)
        loss_enc = F.mse_loss(reconstruct_out, H)
        # HW = self.lin(H)
        # embed = self.emb(HW)
        # embedding = self.embedding(HW)
        
        HW = torch.mm(H, W)
        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            # A = Laplacian_Nonlinear(n, structure, X, m)
            A = Laplacian_Nonlinear(n, structure, hidden_1dim_out, m)
        else: A = structure

        if self.cuda: A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b, loss_enc
    
    def forward_labelencoder(self, structure, H, m=True):


        W, b = self.W, self.bias
        # HW = torch.mm(H, W)
        pred_out, hidden_1dim_out = self.labelencoder(H)
        loss_enc = F.nll_loss(reconstruct_out, H)
        # HW = self.lin(H)
        # embed = self.emb(HW)
        # embedding = self.embedding(HW)
        
        HW = torch.mm(H, W)
        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            # A = Laplacian_Nonlinear(n, structure, X, m)
            A = Laplacian_Nonlinear(n, structure, hidden_1dim_out, m)
        else: A = structure

        if self.cuda: A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b, loss_enc
    

    def forward_coordinate(self, structure, H, m=True):
        W, b = self.W, self.bias

        n, nclos = H.shape[0], H.shape[1]

        if self.cuda: structure = structure.cuda()

        
        # for i in range(len(self.mul_L_real)): # [K, B, N, D]
        #     future.append(torch.jit.fork(process, 
        #                     self.mul_L_real[i], self.mul_L_imag[i], 
        #                     self.weight[i], X_real, X_imag))
        # result = []
        # for i in range(len(self.mul_L_real)):
        #     result.append(torch.jit.wait(future[i]))
        # result = torch.sum(torch.stack(result), dim=0)
        
        future = []
        if self.reapproximate:
            for i in tqdm(range(nclos)): 
                vec = H[:,i]
                future.append(torch.jit.fork(Laplacian_Nonlinear_coordinate, 
                            n, structure, vec, m))

            result = []
            for i in range(nclos):
                result.append(torch.jit.wait(future[i]))

            concat = torch.stack(result, dim =1).squeeze()
        else: concat = structure

        # HW = torch.mm(H, W)

        # if self.cuda: A = A.cuda()
        # A = Variable(A)

        AHW = SparseMM.apply(concat, W)     
        return AHW + b



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2

def nonlinear_laplacian_sparse(row, col, size, q = 0.25, norm = True, laplacian = True, max_eigen = 2, 
gcn_appr = False, edge_weight = None):
    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight, (row, col)), shape=(size, size), dtype=np.float32)
    
    diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
    if gcn_appr:
        A += diag

    A_sym = 0.5*(A + A.T) # symmetrized adjacency

    if norm:
        d = np.array(A_sym.sum(axis=0))[0] # out degree
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:
        Theta = 2*np.pi*q*1j*(A - A.T) # phase angle array
        Theta.data = np.exp(Theta.data)
        if norm:
            D = diag
        else:
            d = np.sum(A_sym, axis = 0) # diag of degree array
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - Theta.multiply(A_sym) #element-wise

    if norm:
        L = (2.0/max_eigen)*L - diag

    return L

def Laplacian_Nonlinear(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    # p = X
    # rv = np.random.rand(X.shape[1])
    rv = np.load("rv_0.npy",allow_pickle=True)
    print(rv)
    print(rv.shape)
    # print("this is random projection: " , rv)
    # np.save("rv_2.npy",rv)
    p = np.dot(X, rv)   #projection onto a random vector rv
    # print(p)
    # p = X

    row, col = E[0], E[1]
    size = V


    Adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    # A_sym = 0.5*(A + A.T) # symmetrized adjacency
    G_undirected = nx.from_scipy_sparse_array(Adj)
    G = nx.from_scipy_sparse_array(Adj,create_using=nx.DiGraph)
    assert G.is_directed()
    directed_edges = [e for e in G.edges]
    diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)

    for k in directed_edges:
        u,v = k[0], k[1]
        if p[u] >= p[v]:
            edges.extend([[u, v], [v, u]])
            if (u,v) not in weights:
                weights[(u,v)] = 0
            weights[(u,v)] += float(1)

            if (v,u) not in weights:
                weights[(v,u)] = 0
            weights[(v,u)] += float(1) 
        else: 
            edges.extend([[u, u], [v, v]])
            if (u,u) not in weights:
                weights[(u,u)] = 0
            weights[(u,u)] += float(1)

            if (v,v) not in weights:
                weights[(v,v)] = 0
            weights[(v,v)] += float(1)   
    
    return adjacency(edges, weights, V, m)

def Laplacian_Nonlinear_torch(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])
    p = np.dot(X, rv)   #projection onto a random vector rv

    # p = X.squeeze()
    # rv = torch.rand(X.shape[1]).cuda()
    # p = torch.matmul(X, rv)   #projection onto a random vector rv
    # p = X

    row, col = E[0], E[1]
    size = V

    lst_directed_edges = [(E[0][i].item(),E[1][i].item()) for i in range(E.shape[1])]
    lst_direction_values = torch.stack([p[E[0][i]] - p[E[1][i]] for i in range(E.shape[1])])
    torch_zeros = torch.zeros(E.shape[1]).cuda()

    max_values = torch.maximum(lst_direction_values,torch_zeros)

    # Adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    # # A_sym = 0.5*(A + A.T) # symmetrized adjacency
    # G_undirected = nx.from_scipy_sparse_matrix(Adj)
    # G = nx.from_scipy_sparse_matrix(Adj,create_using=nx.DiGraph)
    # assert G.is_directed()
    # directed_edges = [e for e in G.edges]
    # diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)

    for i, val in enumerate(max_values):
        u,v = lst_directed_edges[i]
        if val == 0: 
            edges.extend([[u, u], [v, v]])
            if (u,u) not in weights:
                weights[(u,u)] = 0
            weights[(u,u)] += float(1)

            if (v,v) not in weights:
                weights[(v,v)] = 0
            weights[(v,v)] += float(1)
        else: 
            edges.extend([[u, v], [v, u]])
            if (u,v) not in weights:
                weights[(u,v)] = 0
            weights[(u,v)] += float(1)

            if (v,u) not in weights:
                weights[(v,u)] = 0
            weights[(v,u)] += float(1) 
    
    return adjacency(edges, weights, V, m)

def multiprocess(max_values, lst_directed_edges, edges, weights): 
        for i, val in enumerate(max_values):
            u,v = lst_directed_edges[i]
            if val == 0: 
                edges.extend([[u, u], [v, v]])
                if (u,u) not in weights:
                    weights[(u,u)] = 0
                weights[(u,u)] += float(1)

                if (v,v) not in weights:
                    weights[(v,v)] = 0
                weights[(v,v)] += float(1)
            else: 
                edges.extend([[u, v], [v, u]])
                if (u,v) not in weights:
                    weights[(u,v)] = 0
                weights[(u,v)] += float(1)

                if (v,u) not in weights:
                    weights[(v,u)] = 0
                weights[(v,u)] += float(1)

        return (edges, weights) 

# @torch.no_grad()
def Laplacian_Nonlinear_coordinate(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    # rv = np.random.rand(X.shape[1])
    # p = np.dot(X, rv)   #projection onto a random vector rv
    p = X

    row, col = E[0], E[1]
    size = V
    # edges = edges.cuda()
    # weights = weights.cuda()

    lst_directed_edges = [(E[0][i].item(),E[1][i].item()) for i in range(E.shape[1])]
    lst_direction_values = torch.stack([p[E[0][i]] - p[E[1][i]] for i in range(E.shape[1])])
    torch_zeros = torch.zeros(E.shape[1]).cuda()

    max_values = torch.maximum(lst_direction_values,torch_zeros)

    for i, val in enumerate(max_values):
        u,v = lst_directed_edges[i]
        if val == 0: 
            edges.extend([[u, u], [v, v]])
            if (u,u) not in weights:
                weights[(u,u)] = 0
            weights[(u,u)] += float(1) + val 

            if (v,v) not in weights:
                weights[(v,v)] = 0
            weights[(v,v)] += float(1)
        else: 
            edges.extend([[u, v], [v, u]])
            if (u,v) not in weights:
                weights[(u,v)] = 0
            weights[(u,v)] += float(1) + val 

            if (v,u) not in weights:
                weights[(v,u)] = 0
            weights[(v,u)] += float(1) 

    # multi_proceses = Pool(processes=4)
    # result = multi_proceses.map(multiprocess, (max_values, lst_directed_edges, edges, weights))
    # edges, weights  = result[0], result[1]

    Adj = adjacency(edges, weights, V, m).cuda()
    AdjVec = torch.mm(Adj, torch.unsqueeze(X,1))
    
    return AdjVec

'''
    for i, val in enumerate(max_values):
        u,v = lst_directed_edges[i]
        if val == 0: 
            edges.extend([[u, u], [v, v]])
            if (u,u) not in weights:
                weights[(u,u)] = 0
            weights[(u,u)] += float(1)

            if (v,v) not in weights:
                weights[(v,v)] = 0
            weights[(v,v)] += float(1)
        else: 
            edges.extend([[u, v], [v, u]])
            if (u,v) not in weights:
                weights[(u,v)] = 0
            weights[(u,v)] += float(1)

            if (v,u) not in weights:
                weights[(v,u)] = 0
            weights[(v,u)] += float(1) 


    # Adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)

    # # A_sym = 0.5*(A + A.T) # symmetrized adjacency
    # G_undirected = nx.from_scipy_sparse_matrix(Adj)
    # G = nx.from_scipy_sparse_matrix(Adj,create_using=nx.DiGraph)
    # assert G.is_directed()
    # directed_edges = [e for e in G.edges]
    # diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)

    # for k in directed_edges:
    #     u,v = k[0], k[1]
    #     if p[u] >= p[v]:
    #         edges.extend([[u, v], [v, u]])
    #         if (u,v) not in weights:
    #             weights[(u,v)] = 0
    #         weights[(u,v)] += float(1)

    #         if (v,u) not in weights:
    #             weights[(v,u)] = 0
    #         weights[(v,u)] += float(1) 
    #     else: 
    #         edges.extend([[u, u], [v, v]])
    #         if (u,u) not in weights:
    #             weights[(u,u)] = 0
    #         weights[(u,u)] += float(1)

    #         if (v,v) not in weights:
    #             weights[(v,v)] = 0
    #         weights[(v,v)] += float(1) 

'''



def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E.keys():
        hyperedge = list(E[k])
        
        p = np.dot(X[hyperedge], rv)   #projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2*len(hyperedge) - 3    # normalisation constant
        if m:
            
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    
    
    return adjacency(edges, weights, V)

def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights

def adjacency(edges, weights, n,m):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    # if m:
    #     # random.seed(9001) 
    #     alpha = random.random()
    #     print(alpha)
    #     A = rjnormalise(sp.csr_matrix(adj, dtype=np.float32), n,alpha)
    #     GG = nx.from_scipy_sparse_matrix(A)
    #     print(nx.is_connected(GG))
    # else:
    #     # A = sp.csr_matrix(adj, dtype=np.float32)
    #     # A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    #     GG = nx.from_scipy_sparse_matrix(adj)
    #     print(nx.is_connected(GG))
    # # A = ssm2tst(A)
    return adj

def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 

def rjnormalise(M,n,alpha):
    """
    symmetrically random jump normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}

    first_term =  (DHI.dot(M)).dot(DHI)

    dh = np.power(d, 1/2).flatten()
    dh[np.isinf(dh)] = 0.
    DH = sp.diags(dh)
    J = np.ones([n, n], dtype = int)
    sJ =  csr_matrix(J)

    second_term =  (DHI.dot(sJ)).dot(DH)

    random_jump_adj = (alpha * first_term + ((1-alpha)/n) * second_term).astype(np.float)

    # GG = nx.from_scipy_sparse_matrix(random_jump_adj)
    # print(nx.is_connected(GG))


    
    return random_jump_adj

def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)
