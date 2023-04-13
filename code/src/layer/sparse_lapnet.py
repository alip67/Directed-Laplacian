import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np

#internel
from utils import hermitian 
#from torch.nn import MultiheadAttention

def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real, X_real)
    real = torch.matmul(data, weight) 
    data = -1.0*torch.spmm(mul_L_imag, X_imag)
    real += torch.matmul(data, weight) 
    
    data = torch.spmm(mul_L_imag, X_real)
    imag = torch.matmul(data, weight)
    data = torch.spmm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)
    return torch.stack([real, imag])

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def generate_laplcians(X, E, K, row, col, size, q = 0.25, norm = True, laplacian = True, max_eigen = 2, 
gcn_appr = False, m = False, edge_weight = None): 
    try:
        L = hermitian.hermitian_decomp_sparse(X, m,E,row, col, size, q, norm=True, laplacian=laplacian, 
            max_eigen = 2.0, gcn_appr = gcn_appr, edge_weight = edge_weight)
    except AttributeError:
        L = hermitian.hermitian_decomp_sparse(X, m,E,row, col, size, q, norm=True, laplacian=laplacian, 
            max_eigen = 2.0, gcn_appr = gcn_appr, edge_weight = None)

    multi_order_laplacian = hermitian.cheb_poly_sparse(L, K)

    # convert dense laplacian to sparse matrix
    L_img = []
    L_real = []
    for i in range(len(multi_order_laplacian)):
        L_img.append( sparse_mx_to_torch_sparse_tensor(multi_order_laplacian[i].imag).cuda() )
        L_real.append( sparse_mx_to_torch_sparse_tensor(multi_order_laplacian[i].real).cuda() )

    return L_real, L_img

def cheb_poly(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    #multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian = []
    multi_order_laplacian.append(torch.eye(N, requires_grad=True).type(torch.cfloat))
    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian.append(A)
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian.append( 2.0 * A.dot(multi_order_laplacian[k-1]) - multi_order_laplacian[k-2] )

    return multi_order_laplacian

# A for Directed graphs:
class Graph_Directed_A(nn.Module):
      
    def __init__(self, input_dim,structure):
        super(Graph_Directed_A, self).__init__()

        _ , self.Edges, self.K, self.row, self.col,self.size = structure
        self.e1 = nn.Linear(input_dim, 1)
        self.register_parameter('laplcian', None)

    def reset_parameters(self, input):
        self.laplcian = nn.Parameter(input)

    def calc_adjacency(self, edges, weights, n,m):
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

        edges, weights = torch.as_tensor(edges), torch.stack(organised)

        edge_index = torch.stack([edges[:, 0],
                           edges[:, 1]]).cuda()
        
        adj = SparseTensor.from_edge_index(edge_index=edge_index, edge_attr=weights)

        # indices =  torch.vstack((edges[:, 0], edges[:, 1])).type(torch.int64).cuda()
        # # values = torch.ones(len(row))
        # shape = torch.Size((n, n))

        # adj = torch.sparse.FloatTensor(indices, weights, shape)
        # adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)

        # diag = coo_matrix( (np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)

        # indices_eye =  torch.vstack((torch.arange(n), torch.arange(n))).type(torch.int64)
        # values_eye = torch.ones(n)
        # shape = torch.Size((n, n))

        # diag_eye = torch.sparse.FloatTensor(indices_eye, values_eye, shape).cuda()

        diag_eye = SparseTensor.eye(n).cuda()
 
        adj = adj + diag_eye
        return adj

    def Laplacian_Nonlinear_torch(self, V, E, X, m):
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
        p = X.squeeze()

        row, col = E[0], E[1]
        size = V

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
                weights[(u,u)] += float(1)+val

                if (v,v) not in weights:
                    weights[(v,v)] = 0
                weights[(v,v)] += float(1)+val
            else: 
                edges.extend([[u, v], [v, u]])
                if (u,v) not in weights:
                    weights[(u,v)] = 0
                weights[(u,v)] += float(1)+val

                if (v,u) not in weights:
                    weights[(v,u)] = 0
                weights[(v,u)] += float(1)+val 
        
        return self.calc_adjacency(edges, weights, V, m)

    def calculate_new_laplcians(self, X, E, K, row, col, size, q = 0.25, norm = True, laplacian = True, max_eigen = 2, 
            gcn_appr = False, m = False, edge_weight = None):
        
        # indices =  torch.vstack((row, col)).astype(np.int64)
        # values = torch.ones(len(row))
        # shape = torch.Size((size, size))

        # A = torch.sparse.FloatTensor(indices, values, shape)

        A = self.Laplacian_Nonlinear_torch(size, E, X, m)

        A = SparseTensor.to_dense(A) 
        
        diag = torch.eye(size).cuda()
        if gcn_appr:
            A += diag

        # A = Laplacian_Nonlinear(size, E, X, m)

        A_sym = 0.5*(A + A.t()) # symmetrized adjacency

        if norm:
            d = A_sym.sum(axis=0) # out degree
            # d = torch.sparse.sum(A_sym, dim = 0).to_dense()
            d[d == 0] = 1
            d = torch.pow(d, -0.5)

            # D = torch.sparse.FloatTensor(torch.vstack((torch.arange(size), torch.arange(size))).type(torch.int64).cuda(), d, torch.Size((size, size)))
            
            edge_index = torch.vstack((torch.arange(size), torch.arange(size))).type(torch.int64).cuda()
        
            D = SparseTensor.from_edge_index(edge_index=edge_index, edge_attr=d)

            # D  = torch.sparse.spdiags(d, torch.tensor([0]), (size, size))
            # D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
            # A_sym = D.dot(A_sym).dot(D)
            # A_sym = torch.sparse.mm(D,torch.sparse.mm(A_sym,D))
            A_sym = d.unsqueeze(1) * A_sym * d


        if laplacian:
            Theta = (2*torch.tensor(math.pi)*q*1j*(A - A.T)) # phase angle array
            Theta.data = torch.exp(Theta.data)
            if norm:
                D = diag
            else:
                d = torch.sum(A_sym, axis = 0) # diag of degree array
                D = torch.sparse.FloatTensor(torch.vstack((torch.arange(size), torch.arange(size))).type(torch.int64).cuda(), d, torch.Size((size, size))).cuda()
                # D  = torch.sparse.spdiags(d, torch.tensor([0]), (size, size)).cuda()
                # D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
            L = D - Theta.multiply(A_sym) #element-wise

        if norm:
            L = (2.0/max_eigen)*L - diag

        multi_order_laplacian = cheb_poly(L, K)

        # convert dense laplacian to sparse matrix
        L_img = []
        L_real = []
        for i in range(len(multi_order_laplacian)):
            L_img.append( multi_order_laplacian[i].imag.cuda() )
            L_real.append( multi_order_laplacian[i].real.cuda() )

        return L_real, L_img

    def forward(self, featuers):
        
        m1 = self.e1(featuers)
        # print(self.e1.weight.grad)
        L_real, L_img = self.calculate_new_laplcians(m1,self.Edges, self.K, self.row, self.col, self.size)
        # if self.laplcian is None:
        #     self.reset_parameters(input)
        
        return L_real, L_img

class LapConv(nn.Module):
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """
    def __init__(self, in_c, out_c, K,structure,reapproximate=False, bias=True):
        super(LapConv, self).__init__()

        # L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # # list of K sparsetensors, each is N by N
        # self.mul_L_real = L_norm_real   # [K, N, N]
        # self.mul_L_imag = L_norm_imag   # [K, N, N]

        # self.mul_L_real = None   # [K, N, N]
        # self.mul_L_imag = None   # [K, N, N]

        self.recompute = reapproximate


        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        if reapproximate:
            self.graph_struct = Graph_Directed_A(in_c,structure)
            self.mul_L_real = None   # [K, N, N]
            self.mul_L_imag = None   # [K, N, N]
        else:
            self.mul_L_real = None   # [K, N, N]
            self.mul_L_imag = None   # [K, N, N]
            # self.register_parameter("mul_L_real", None)
            # self.register_parameter("mul_L_imag", None)

    def forward(self, data):
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]

        real = 0.0
        imag = 0.0

        future = []
        for i in range(len(self.mul_L_real)): # [K, B, N, D]
            future.append(torch.jit.fork(process, 
                            self.mul_L_real[i], self.mul_L_imag[i], 
                            self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(self.mul_L_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias

class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real, img):
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img

class LapNet(nn.Module):
    def __init__(self, in_c, dataset, q, num_filter=2, K=2, label_dim=2, layer=2, dropout=False, fast=True):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian

        # self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, layer=2, dropout=False
        """
        super(LapNet, self).__init__()

        activation_func = complex_relu_layer

        # self.linear_layer = torch.nn.Linear(in_c, 1)
        
        # fast = True

        if fast:
            self.reapproximate = False
            size = dataset[0].y.size(-1)
            #adj = torch.zeros(size, size).data.numpy().astype('uint8')
            #adj[dataset[0].edge_index[0], dataset[0].edge_index[1]] = 1

            f_node, e_node = dataset[0].edge_index[0], dataset[0].edge_index[1]

            label = dataset[0].y.data.numpy().astype('int')
            X = dataset[0].x.data.numpy().astype('float32')
            E = dataset[0].edge_index
            L_norm_real,L_norm_imag = generate_laplcians(X, E, K, f_node, e_node, size, q, norm=True, laplacian=True, 
            max_eigen = 2.0, gcn_appr = False, m= False, edge_weight = dataset[0].edge_weight)
            self.L_real, self.L_imag  = L_norm_real , L_norm_imag
            self.structure= None

            chebs = [LapConv(in_c, num_filter, K,self.structure, self.reapproximate)]
            chebs.append(activation_func())

            for i in range(1, layer):
                chebs.append(LapConv(num_filter, num_filter, K,self.structure, self.reapproximate))
                chebs.append(activation_func())

            self.Chebs = torch.nn.Sequential(*chebs)
      
        else:
            self.reapproximate = True
            size = dataset[0].y.size(-1)

            f_node, e_node = dataset[0].edge_index[0], dataset[0].edge_index[1]

            label = dataset[0].y.data.numpy().astype('int')
            X = dataset[0].x.data.numpy().astype('float32')
            E = dataset[0].edge_index
            self.register_parameter('L_real', None)
            self.register_parameter('L_imag', None)
            self.structure = (torch.tensor(X), E, K, f_node, e_node,size)

            self.cheb_lapConv1 = LapConv(in_c, num_filter, K,self.structure, self.reapproximate)
            self.chebactive1 = activation_func()

            self.cheb_lapConv2 = LapConv(num_filter, num_filter, K,self.structure, self.reapproximate)
            self.chebactive2 = activation_func()

        # chebs = [LapConv(in_c, num_filter, K,self.structure, self.reapproximate)]
        # chebs.append(activation_func())

        # self.cheb_lapConv1 = LapConv(in_c, num_filter, K,self.structure, self.reapproximate)
        # self.chebactive1 = activation_func()

        # for i in range(1, layer):
        #     chebs.append(LapConv(num_filter, num_filter, K,self.structure, self.reapproximate))
        #     chebs.append(activation_func())

        # self.Chebs = torch.nn.Sequential(*chebs)

        # self.cheb_lapConv2 = LapConv(num_filter, num_filter, K,self.structure, self.reapproximate)
        # self.chebactive2 = activation_func()

        last_dim = 2  
        self.Conv = nn.Conv1d(num_filter*last_dim, label_dim, kernel_size=1)        
        self.dropout = dropout

    # def colculate_embedding(self, real, img):

    #     return None

    def forward(self, real, imag):
        if self.reapproximate:
            l_real, l_imag = self.cheb_lapConv1.graph_struct(real)
            self.cheb_lapConv1.mul_L_real = l_real
            self.cheb_lapConv1.mul_L_imag = l_imag

            real_lapconv0, imag_lapconv0 = self.cheb_lapConv1((real, imag))
            
            realchebactive1, imagchebactive1 = self.chebactive1(real_lapconv0,imag_lapconv0)
            
            l_real1, l_imag1 = self.cheb_lapConv2.graph_struct(realchebactive1)
            self.cheb_lapConv2.mul_L_real = l_real1
            self.cheb_lapConv2.mul_L_imag = l_imag1
            
            real_lapconv1, imag_lapconv1 = self.cheb_lapConv2((realchebactive1, imagchebactive1))
            
            real, imag = self.chebactive1(real_lapconv1,imag_lapconv1)
        else:
            self.Chebs[0].mul_L_real = self.L_real
            self.Chebs[0].mul_L_imag = self.L_imag
            self.Chebs[2].mul_L_real = self.L_real
            self.Chebs[2].mul_L_imag = self.L_imag
            real, imag = self.Chebs((real, imag))
        # real, imag = self.Chebs((real, imag))
        x = torch.cat((real, imag), dim = -1)
        
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return x

# # class MagNet_Edge(nn.Module):
#     def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim = 2, layer = 2, dropout = False):
#         super(MagNet_Edge, self).__init__()
        
#         activation_func = complex_relu_layer

#         chebs = [MagConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
#         chebs.append(activation_func())

#         for i in range(1, layer):
#             chebs.append(MagConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
#             chebs.append(activation_func())
#         self.Chebs = torch.nn.Sequential(*chebs)
        
#         last_dim = 2
#         self.linear = nn.Linear(num_filter*last_dim*2, label_dim)     
#         self.dropout = dropout

#     def forward(self, real, imag, index):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real[index[:,0]], real[index[:,1]], imag[index[:,0]], imag[index[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x