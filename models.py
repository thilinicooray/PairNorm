from layers import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class SGC(nn.Module):
    # for SGC we use data without normalization
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, norm_mode='None', norm_scale=10, **kwargs):
        super(SGC, self).__init__()
        self.linear = torch.nn.Linear(nfeat, nclass)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.dropout = nn.Dropout(p=dropout)
        self.nlayer = nlayer      
        
    def forward(self, x, adj):
        x = self.norm(x)
        for _ in range(self.nlayer):
            x = adj.mm(x)
            x = self.norm(x)  
        x = self.dropout(x)
        x = self.linear(x)
        return x 
        
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)

        #vae
        self.gc_mu_adj = GraphConv(nhid, nhid)
        self.gc_logvar_adj = GraphConv(nhid, nhid)

        self.gc_mu_node = GraphConv(nhid, nhid)
        self.gc_logvar_node = GraphConv(nhid, nhid)

        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.node_regen = GraphConv(nhid, nhid+nfeat)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)

    def encode(self, hidden1, adj, gc2, gc3, gc4, gc5):
        return self.relu(gc2(hidden1, adj)), self.relu(gc3(hidden1, adj)), self.relu(gc4(hidden1, adj)), self.relu(gc5(hidden1, adj))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x_org, adj):
        x = self.dropout(x_org)
        x = self.relu(self.gc1(x, adj))

        #vae
        mu, logvar, mu_n, var_n = self.encode(self.dropout(x), adj, self.gc_mu_adj, self.gc_logvar_adj, self.gc_mu_node, self.gc_logvar_node)
        z = self.reparameterize(mu, logvar)
        z_n = self.reparameterize(mu_n, var_n)
        adj1 = self.dc(z)


        #get masked new adj
        zero_vec = torch.sparse.FloatTensor(-9e15*torch.ones_like(adj1))
        masked_adj = torch.where(adj > 0, adj1, zero_vec)
        #masked_adj = adj * adj1 + zero_vec
        adj1 = F.softmax(masked_adj, dim=1)

        a1 = self.relu(self.node_regen(self.dropout(z_n), adj1.t()))
        zero_vec = -9e15*torch.ones_like(a1)
        masked_nodes = torch.where(x_org > 0, a1, zero_vec)
        a1 = F.softmax(masked_nodes, dim=1)

        x = torch.cat([a1, x], -1)
        #x = self.norm(x)
        x = self.dropout(x)
        x = self.gc2(x, adj+adj1)
        return a1, adj1 , mu, logvar, mu_n, var_n, x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhead, 
                 norm_mode='None', norm_scale=1,**kwargs):
        super(GAT, self).__init__()
        alpha_droprate = dropout
        self.gac1 = GraphAttConv(nfeat, nhid, nhead, alpha_droprate)
        self.gac2 = GraphAttConv(nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True) 
        self.norm = PairNorm(norm_mode, norm_scale)

    def forward(self, x, adj):
        x = self.dropout(x) # ?
        x = self.gac1(x, adj)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gac2(x, adj)
        return x

class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGCN, self).__init__()
        assert nlayer >= 1 
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i==0 else nhid, nhid) 
            for i in range(nlayer-1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer==1 else nhid , nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip>0 and i%self.skip==0:
                x = x + x_old
                x_old = x
            
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x

class DeepGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayer=2, residual=0, nhead=1,
                 norm_mode='None', norm_scale=1, **kwargs):
        super(DeepGAT, self).__init__()
        assert nlayer >= 1 
        alpha_droprate = dropout
        self.hidden_layers = nn.ModuleList([
            GraphAttConv(nfeat if i==0 else nhid, nhid, nhead, alpha_droprate)
            for i in range(nlayer-1)
        ])
        self.out_layer = GraphAttConv(nfeat if nlayer==1 else nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.skip = residual

    def forward(self, x, adj):
        x_old = 0
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.norm(x)
            x = self.relu(x)
            if self.skip>0 and i%self.skip==0:
                x = x + x_old
                x_old = x
                
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj