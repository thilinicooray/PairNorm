import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(preds, labels, mu, logvar, n_nodes):
    cost = F.binary_cross_entropy_with_logits(preds, labels)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def train(net, optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    noderegen, recovered, mu, logvar, mu_n, var_n, output = net(data.x, data.adj)
    nc_loss = criterion(output[data.train_mask], data.y[data.train_mask])
    ae_loss = loss_function(preds=recovered[data.train_mask], labels=data.adj[data.train_mask],
                            mu=mu[data.train_mask], logvar=logvar[data.train_mask], n_nodes=data.adj.size[0])
    node_ae_loss = loss_function(preds=noderegen[data.train_mask], labels=data.x[data.train_mask],
                                 mu=mu_n[data.train_mask], logvar=var_n[data.train_mask], n_nodes=data.adj.size[0])
    loss = nc_loss + 0.1*ae_loss + 0.1*node_ae_loss
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, acc 

def val(net, criterion, data):
    net.eval()
    output = net(data.x, data.adj)
    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val

def test(net, criterion, data):
    net.eval()
    output = net(data.x, data.adj)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)