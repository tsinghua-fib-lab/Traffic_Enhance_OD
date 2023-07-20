import os
import json
from turtle import forward
from xml.dom.pulldom import DOMEventStream

import numpy as np
from sklearn.utils import indices_to_mask

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn import GraphConv, GATConv

class GNN(nn.Module):
    def __init__(self, config, out_dim):
        super(GNN, self).__init__()
        self.conv_in = GraphConv(config["GNN_in_dim"], config["GNN_hid_dim"])
        self.conv_hid = GraphConv(config["GNN_hid_dim"], config["GNN_hid_dim"])
        self.conv_out = GraphConv(config["GNN_hid_dim"], out_dim)

    def forward(self, g, x):
        h = torch.relu(self.conv_in(g, x))
        h = torch.relu(self.conv_hid(g, h))
        out = self.conv_out(g, h)
        return out

class bilinear_predictor(nn.Module):
    def __init__(self, config):
        super(bilinear_predictor, self).__init__()
        self.config = config
        self.bilinear_od = nn.Bilinear(config["node_embsize"], config["node_embsize"], config["node_embsize"])
        self.bilinear_dis = nn.Bilinear(config["node_embsize"], config["disProj_dim"], 1)

    def forward(self, ori, dst, dis):
        od_emb = torch.relu(self.bilinear_od(ori, dst))
        flow = torch.tanh(self.bilinear_dis(od_emb, dis))
        return flow

class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer):
        super(ResMLP, self).__init__()
        self.num_layer = num_layer
        self.linear_in = nn.Linear(in_dim, out_dim)

        self.linears = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for i in range(num_layer)]
        )

    def forward(self, input):
        input = self.linear_in(input)
        mid = input
        for i in range(self.num_layer):
            mid = torch.relu(self.linears[i](mid)) # + mid
        output = mid + input
        return output

class ResMLP_predictor(nn.Module):
    def __init__(self, config):
        super(ResMLP_predictor, self).__init__()
        if config["IF_disProj"] == 0:
            self.mlp = ResMLP(config["node_embsize"]*2+1, config["node_embsize"], config["num_mlp_layers"])
        else:
            self.mlp = ResMLP(config["node_embsize"]*2+config["disProj_dim"], config["node_embsize"], config["num_mlp_layers"])

        self.linear_out = nn.Linear(config["node_embsize"], 1)

    def forward(self, input):
        output = self.mlp(input)
        flow = torch.tanh(self.linear_out(output))
        return flow

class linear_predictor(nn.Module):
    def __init__(self, config):
        super(linear_predictor, self).__init__()
        self.config = config
        
        self.linear = nn.Linear(config["node_embsize"] * 2 + 1, 1)

    def forward(self, features):
        flow = torch.tanh(self.linear(features))
        return flow

class xy_projector(nn.Module):
    def __init__(self, config):
        super(xy_projector, self).__init__()

        self.linear_in = nn.Linear(2, config["locProj_hid_dim"])

        self.linears = nn.ModuleList(
            [nn.Linear(config["locProj_hid_dim"], config["locProj_hid_dim"]) for i in range(config["num_locProj_layers"])]
        )

    def forward(self, xy):
        hid = self.linear_in(xy)
        for layer in self.linears:
            hid = torch.relu(layer(hid))
        return hid

class dis_projector(nn.Module):
    def __init__(self, config):
        super(dis_projector, self).__init__()

        self.linear_in = nn.Linear(1, config["disProj_dim"])

        self.linears = nn.ModuleList(
            [nn.Linear(config["disProj_dim"], config["disProj_dim"]) for i in range(config["disProj_layers"])]
        )

    def forward(self, dis):
        dis = self.linear_in(dis)
        mid = dis
        for layer in self.linears:
            mid = torch.relu(layer(mid))
        out = mid
        return out

class rnn_with_graph_convolution(nn.Module):
    def __init__(self, config):
        super(rnn_with_graph_convolution, self).__init__()
        
        self.graph_conv_ru = GNN(config, config["node_embsize"] * 2)
        
        self.graph_conv_reset_h = GNN(config, config["node_embsize"])

    def forward(self, g, Xs, h0 = torch.zeros([500, 64])):
        h0 = h0.to(Xs.device)
        hs = []
        h = h0
        for idx, x in enumerate(Xs):
            # compute resetting and updating weights
            ru = torch.sigmoid(self.graph_conv_ru(g, torch.cat((h, x), dim=-1)))
            r, z =  torch.split(tensor=ru, split_size_or_sections=(ru.size(-1)//2), dim=-1)

            # reset h
            h_ = h * r
            h_ = torch.tanh(self.graph_conv_reset_h(g, torch.cat((h_, x), dim=-1)))
            
            # update
            h = z * h + (1 - z) * h_
            hs.append(h)
        hs = torch.stack(hs)
        return hs

class ST_transor(nn.Module):
    def __init__(self, config):
        super(ST_transor, self).__init__()
        self.config = config

        if config["IF_locProj"] == 1:
            self.loc_projector = xy_projector(config)

        self.RNN_with_GraphConv = rnn_with_graph_convolution(config)

        if config["predictor_type"] == "bilinear":
            self.flow_predictor = bilinear_predictor(config)
        elif config["predictor_type"] == "linear":
            self.flow_predictor = linear_predictor(config)
        elif config["predictor_type"] == "ResMLP":
            self.flow_predictor = ResMLP_predictor(config)
        
        if self.config["IF_disProj"] == 1:
            self.dis_projector = dis_projector(config)

    def forward(self, g, x, dis, train_idx):
        # spatial temporal graph nn
        feat = x[:, :, :-2]
        node_emb_24 = self.RNN_with_GraphConv(g, feat, h0=torch.zeros([x.size(1), self.config["node_embsize"]]))
        ori_emb = node_emb_24[:, train_idx[0], :]
        dst_emb = node_emb_24[:, train_idx[1], :]

        # prediction based on node embedddings
        if self.config["predictor_type"] == "bilinear":
            dis = dis[train_idx]
            dis = dis.unsqueeze(dim=0).unsqueeze(dim=2)
            dis = dis.repeat([24, 1, 1])
            if self.config["IF_disProj"] == 1:
                dis = self.dis_projector(dis)
            flows = self.flow_predictor(ori_emb, dst_emb, dis)
        else:
            dis = dis[train_idx]
            dis = dis.unsqueeze(dim=0).unsqueeze(dim=2)
            dis = dis.repeat([24, 1, 1])
            if self.config["IF_disProj"] == 1:
                dis = self.dis_projector(dis)
            od_emb = torch.cat((ori_emb, dst_emb, dis), dim=-1)
            flows = self.flow_predictor(od_emb)

        return flows.squeeze().transpose(1, 0)

class bipartite_GNN(nn.Module):
    def __init__(self, config):
        super(bipartite_GNN, self).__init__()

        self.linearIn = nn.Linear(1, config["pre_hid"])
        self.linearAll = nn.Linear(config["pre_len"] * config["pre_hid"], config["pre_hid"])
        self.attn = nn.MultiheadAttention(config["pre_hid"], 6)
        
        self.g_conv_in = GATConv(config["bipartite_in_dim"], config["bipartite_hid_dim"], config["head_num"])
        self.g_conv_hid = GATConv(config["bipartite_hid_dim"], config["bipartite_hid_dim"], config["head_num"])
        self.g_conv_out = GATConv(config["bipartite_hid_dim"], config["bipartite_out_dim"], config["head_num"])

    def forward(self, g, x):

        x = self.linearIn(x)
        all = self.linearAll(x)
        x = torch.relu(self.attn(all, x, x))

        x = torch.relu(self.g_conv_in(g, x))
        x = torch.relu(self.g_conv_hid(g, x))
        x = torch.relu(self.g_conv_out(g, x))

        return None
    
class combined(nn.Module):
    def __init__(self, config):
        super(combined, self).__init__()

        self.ODpart = ST_transor(config)
        self.Trafpart = bipartite_GNN(config)
    
    def forward(self, g1, g2, x, dis, train_idx):
        x = self.ODpart(g1, x, dis, train_idx)
        x = self.Trafpart(g2, x)
        return x






if __name__ == "__main__":
    
    with open("/data/rongcan/code/24h-OD/src/config/beijing.json") as f:
        config = json.load(f)

    device = torch.device(config["device"])
    print(device)

    node_feats_24 = torch.randn([24, 500, 250 + 24]).to(device)
    config["GNN_in_dim"] = node_feats_24.size(-1) + config["node_embsize"]
    print(node_feats_24.size())
    exit()
    print(config)

    g = torch.rand(500, 500).numpy()
    g[g > 0.8] = 1
    g[g<= 0.8] = 0
    g = dgl.graph(g.nonzero()).to(device)
    print(g)

    dis = torch.rand([500, 500]).to(device)
    print(dis.size())

    OD = np.random.randint(low=0, high=100000, size=[24, 500, 500])
    OD = OD / 10000
    OD = OD.astype(np.int)
    print(OD.shape)

    from random import sample
    train_idx = OD.mean(0).nonzero()
    train_sub = sample(list(range(len(train_idx[0]))), 100000)
    train_idx = (train_idx[0][train_sub], train_idx[1][train_sub])

    model = ST_transor(config)
    model = model.to(device)
    print(model)

    model(g, node_feats_24, dis, train_idx)
    
    

