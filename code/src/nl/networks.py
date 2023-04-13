import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch.autograd import Variable
from nl import utilsnl 



class HyperGCN(nn.Module):
    def __init__(self, V, E, X, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = args.d, args.layer, args.c
        cuda = torch.cuda.is_available()

        # self.embedding = nn.Linear(d, 1, bias=True)
        # self.activation = nn.ReLU()

        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if args.dataset == 'citeseer': power = l - i + 4
            h.append(2**power)
        h.append(c)

        if args.fast:
            reapproximate = False
            structure = utilsnl.Laplacian_Nonlinear(V, E, X, args.mediators)        
        else:
            reapproximate = True
            structure = E #edge
            
        self.layers = nn.ModuleList([utilsnl.HyperGraphConvolution(h[i], h[i+1], X.shape[1],reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, args.layer
        self.structure, self.m = structure, args.mediators
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # torch.nn.init.xavier_uniform_(self.embedding.weight)
    #     self.embedding.reset_parameters()




    def forward(self, H,label,train_idx):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m

        total_loss_enc = 0

        # x = self.embedding(H)
        # x = self.activation(x)
        
        for i, hidden in enumerate(self.layers):
            # gcn_pred, enc_loss=hidden.forward_autoencoder(self.structure, H, m)
            # H = F.relu(gcn_pred)
            # total_loss_enc += enc_loss

            # gcn_pred, enc_loss=hidden.forward_labelencoder(self.structure, H, m)
            # H = F.relu(gcn_pred)
            # total_loss_enc += enc_loss
            H = F.relu(hidden.forward_coordinate(self.structure, H, m))
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)
        
        return F.log_softmax(H, dim=1),total_loss_enc
