import os
import time
from turtle import forward
from tqdm import tqdm
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

import torch
import torch.nn as nn
from torch.optim import Adam

from utils.metrics import *
from models.ours import GNN

class one_gnn_for_onehour(nn.Module):
    def __init__(self, config):
        super(one_gnn_for_onehour, self).__init__()

        self.gnn = GNN(config, config["node_embsize"])
        self.linear_out = nn.Linear(config["node_embsize"]*2+1, 1)

    def forward(self, g, x, dis, train_idx):
        g_emb = torch.sigmoid(self.gnn(g, x))
        o_emb = g_emb[train_idx[0]]
        d_emb = g_emb[train_idx[1]]
        dis_emb = dis[train_idx].reshape([-1, 1])
        emb = torch.cat((o_emb, d_emb, dis_emb), dim=1)
        pre = torch.tanh(self.linear_out(emb))
        return pre

class one_gnn(nn.Module):
    def __init__(self, config):
        super(one_gnn, self).__init__()

        self.gnn = one_gnn_for_onehour(config)

    def forward(self, g, x , dis, train_idx):
        pres = []
        for i in range(24):
            pre = self.gnn(g, x[i], dis, train_idx).squeeze()
            pres.append(pre)
        pres = torch.stack(pres, dim=1)
        return pres

class multi_gnn(nn.Module):
    def __init__(self, config):
        super(multi_gnn, self).__init__()

        self.gnns = nn.ModuleList(
            [one_gnn_for_onehour(config) for i in range(24)]
        )
            
    def forward(self, g, x, dis, train_idx):
        x = x[:, :, :-24]
        pres = []
        for i in range(24):
            pre = self.gnns[i](g, x[i], dis, train_idx).squeeze()
            pres.append(pre)
        pres = torch.stack(pres, dim=1)
        return pres

class gravity(nn.Module):
    def __init__(self, config):
        super(gravity, self).__init__()
        self.config = config

        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.log(x)
        logy = self.linear(x)
        y = torch.exp(logy)
        return y

class single_gravity(nn.Module):
    def __init__(self, config):
        super(single_gravity, self).__init__()
        self.config = config

        self.model = gravity(config)
    
    def forward(self, x):
        x = x + 1e-10
        pres = []
        for i in range(24):
            pre = self.model(x)
            pres.append(pre)
        pres = torch.stack(pres)
        return pres.squeeze().transpose(0, 1)

class multi_gravity(nn.Module):
    def __init__(self, config):
        super(multi_gravity, self).__init__()
        self.config = config

        self.models = nn.ModuleList(
            [gravity(config) for x in range(24)]
        )
    
    def forward(self, x):
        x = x + 1e-10
        pres = []
        for model in self.models:
            pre = model(x)
            pres.append(pre)
        pres = torch.stack(pres)
        return pres.squeeze().transpose(0, 1)

class DeepGravity(nn.Module):
    def __init__(self, config):
        super(DeepGravity, self).__init__()
        self.config = config

        self.linear_in = nn.Linear(495, 64)
        self.linears = nn.ModuleList(
            [nn.Linear(64, 64) for i in range(5)]
        )
        self.linear_out = nn.Linear(64, 1)

    def forward(self, input):
        input = self.linear_in(input)
        x = input
        for layer in self.linears:
            x = torch.relu(layer(x))
        x = x + input
        x = torch.tanh(self.linear_out(x))
        return x

class one_DG(nn.Module):
    def __init__(self, config):
        super(one_DG, self).__init__()
        self.config = config

        self.model = DeepGravity(config)

    def forward(self, x):
        pres = []
        for i in range(24):
            pre = self.model(x)
            pres.append(pre.squeeze())
        pres = torch.stack(pres)
        return pres.transpose(0, 1)

class multi_DG(nn.Module):
    def __init__(self, config):
        super(multi_DG, self).__init__()
        self.config = config

        self.models = nn.ModuleList(
            [DeepGravity(config) for i in range(24)]
        )

    def forward(self, x):
        pres = []
        for model in self.models:
            pre = model(x)
            pres.append(pre.squeeze())
        pres = torch.stack(pres)
        return pres.transpose(0, 1)




def multi_rf(config, input, trainDataloader, validDataloader, testDataloader, logger):
    print("---------- multi random forest ----------")
    train_x = []
    train_y = []
    for od_pair_idx, od_flow in trainDataloader:
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.numpy().astype(np.float32).transpose(1, 0).reshape([-1, 1])
        o_feats = input["region_node_feats"][:, od_pair_idx[0], :].cpu().numpy().astype(np.float32).reshape([-1, 271])
        d_feats = input["region_node_feats"][:, od_pair_idx[1], :].cpu().numpy().astype(np.float32).reshape([-1, 271])
        dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32).reshape([1, -1, 1])
        dis_feats = np.repeat(dis_feats, 24, axis=0).reshape([-1, 1])
        feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
        train_x.append(feats)
        train_y.append(od_flow)
    train_x_lastbatch = train_x[-1]
    train_y_lastbatch = train_y[-1]
    train_x = np.stack(train_x[:-1]).reshape([-1, train_x_lastbatch.shape[1]])
    train_y = np.stack(train_y[:-1]).reshape([-1, 1])
    train_x = np.concatenate((train_x, train_x_lastbatch), axis=0).reshape([24, -1, train_x.shape[1]])
    train_y = np.concatenate((train_y, train_y_lastbatch), axis=0).reshape([-1]).reshape([24, -1])

    print("train data: ", train_x.shape, train_y.shape)
    
    # test data
    od_pair_idx, od_flow = next(iter(testDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32).transpose(1, 0)
    o_feats = input["region_node_feats"][:, od_pair_idx[0], :].cpu().numpy().astype(np.float32)
    d_feats = input["region_node_feats"][:, od_pair_idx[1], :].cpu().numpy().astype(np.float32)
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32).reshape([1, -1, 1])
    dis_feats = np.repeat(dis_feats, 24, axis=0)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=2)
    test_x = feats
    test_y = od_flow
    print("test data: ", test_x.shape, test_y.shape)

    # model
    pres = []
    for i in range(24):
        print("for hour", i)
        rf_path = logger.model_directory + "multirf" + str(i) + "_" + config["city"] + ".joblib"
        if os.path.exists(rf_path):
            print("  Load model...")
            randomforest = joblib.load(rf_path)
        else:
            print("  Init model...")
            randomforest = RandomForestRegressor(n_estimators = 20,
                                                 oob_score = True,
                                                 max_depth = None,
                                                 min_samples_split = 10,
                                                 min_samples_leaf = 3,
                                                 n_jobs = cpu_count() // 2)
            # fit
            train_hour_x = train_x[i]
            train_hour_y = train_y[i]
            test_hour_x = test_x[i]
            print("  Fit model...")
            randomforest.fit(X = train_hour_x, y = train_hour_y)
            print("  Save model...")
            joblib.dump(randomforest, rf_path)

        print("Predicting...")
        prediction = randomforest.predict(test_hour_x)
        pres.append(prediction)

    pres = np.stack(pres)
    pre = reMinMax(pres, input["OD_minmax"])
    od_flow = reMinMax(test_y, input["OD_minmax"])
    rmse = float(RMSE_np(pre, od_flow))
    nrmse = float(NRMSE_np(pre, od_flow))
    mae = float(MAE_np(pre, od_flow))
    mape = float(MAPE_np(pre, od_flow))
    smape = float(SMAPE_np(pre, od_flow))
    cpc = float(CPC_np(pre, od_flow))
    print("RMSE = ", rmse)
    print("NRMSE = ", nrmse)
    print("mae = ", mae)
    print("mape = ", mape)
    print("smape = ", smape)
    print("cpc=", cpc)

def random_forest(config, input, trainDataloader, validDataloader, testDataloader, logger):
    print("---------- random forest ----------")
    # train data
    train_x = []
    train_y = []
    for od_pair_idx, od_flow in trainDataloader:
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.numpy().astype(np.float32).transpose(1, 0).reshape([-1, 1])
        o_feats = input["region_node_feats"][:, od_pair_idx[0], :].cpu().numpy().astype(np.float32).reshape([-1, 271])
        d_feats = input["region_node_feats"][:, od_pair_idx[1], :].cpu().numpy().astype(np.float32).reshape([-1, 271])
        dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32).reshape([1, -1, 1])
        dis_feats = np.repeat(dis_feats, 24, axis=0).reshape([-1, 1])
        feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1) # , dis_feats
        train_x.append(feats)
        train_y.append(od_flow)
    train_x_lastbatch = train_x[-1]
    train_y_lastbatch = train_y[-1]
    train_x = np.stack(train_x[:-1]).reshape([-1, train_x_lastbatch.shape[1]])
    train_y = np.stack(train_y[:-1]).reshape([-1, 1])
    train_x = np.concatenate((train_x, train_x_lastbatch), axis=0)
    train_y = np.concatenate((train_y, train_y_lastbatch), axis=0).reshape([-1])
    print("train data: ", train_x.shape, train_y.shape)

    # test data
    od_pair_idx, od_flow = next(iter(testDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32).transpose(1, 0).reshape([-1, 1])
    o_feats = input["region_node_feats"][:, od_pair_idx[0], :].cpu().numpy().astype(np.float32).reshape([-1, 271])
    d_feats = input["region_node_feats"][:, od_pair_idx[1], :].cpu().numpy().astype(np.float32).reshape([-1, 271])
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32).reshape([1, -1, 1])
    dis_feats = np.repeat(dis_feats, 24, axis=0).reshape([-1, 1])
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1) # , dis_feats
    test_x = feats
    test_y = od_flow.reshape([-1])
    print("test data: ", test_x.shape, test_y.shape)

    # model
    rf_path = logger.model_directory + "rf_" + config["city"] + "_" + str(config["random_seed"]) + ".joblib"
    if os.path.exists(rf_path):
        print("Load model...")
        randomforest = joblib.load(rf_path)
    else:
        print("Init model...")
        randomforest = RandomForestRegressor(n_estimators = 20,
                                            oob_score = True,
                                            max_depth = None,
                                            min_samples_split = 10,
                                            min_samples_leaf = 3,
                                            #   verbose=3,
                                            n_jobs = cpu_count() // 2)
        print("model: ", randomforest)

        print("Fit model...")
        start = time.time()
        randomforest.fit(X = train_x, y = train_y)
        print("Save model...")
        joblib.dump(randomforest, rf_path)
        print('Consume ', time.time()-start, ' seconds!')

    print("Predicting...")
    starttime = time.time()
    prediction = randomforest.predict(test_x)
    print(time.time() - starttime)
    exit()

    pre = reMinMax(prediction, input["OD_minmax"])
    od_flow = reMinMax(test_y, input["OD_minmax"])
    rmse = float(RMSE_np(pre, od_flow))
    nrmse = float(NRMSE_np(pre, od_flow))
    mae = float(MAE_np(pre, od_flow))
    mape = float(MAPE_np(pre, od_flow))
    smape = float(SMAPE_np(pre, od_flow))
    cpc = float(CPC_np(pre, od_flow))
    print("RMSE = ", rmse)
    print("NRMSE = ", nrmse)
    print("mae = ", mae)
    print("mape = ", mape)
    print("smape = ", smape)
    print("cpc=", cpc)

def random_forest_noT(config, input, trainDataloader, validDataloader, testDataloader, logger):
    print("---------- random forest no Time Embedding ----------")
    # train data
    train_x = []
    train_y = []
    for od_pair_idx, od_flow in trainDataloader:
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.numpy().astype(np.float32).transpose(1, 0).reshape([-1, 1])
        o_feats = input["region_node_feats"][:, od_pair_idx[0], :].cpu().numpy().astype(np.float32).reshape([-1, 271])[:, :-24]
        d_feats = input["region_node_feats"][:, od_pair_idx[1], :].cpu().numpy().astype(np.float32).reshape([-1, 271])[:, :-24]
        dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32).reshape([1, -1, 1])
        dis_feats = np.repeat(dis_feats, 24, axis=0).reshape([-1, 1])
        feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1) # , dis_feats
        train_x.append(feats)
        train_y.append(od_flow)
    train_x_lastbatch = train_x[-1]
    train_y_lastbatch = train_y[-1]
    train_x = np.stack(train_x[:-1]).reshape([-1, train_x_lastbatch.shape[1]])
    train_y = np.stack(train_y[:-1]).reshape([-1, 1])
    train_x = np.concatenate((train_x, train_x_lastbatch), axis=0)
    train_y = np.concatenate((train_y, train_y_lastbatch), axis=0).reshape([-1])
    print("train data: ", train_x.shape, train_y.shape)

    # test data
    od_pair_idx, od_flow = next(iter(testDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32).transpose(1, 0).reshape([-1, 1])
    o_feats = input["region_node_feats"][:, od_pair_idx[0], :].cpu().numpy().astype(np.float32).reshape([-1, 271])[:, :-24]
    d_feats = input["region_node_feats"][:, od_pair_idx[1], :].cpu().numpy().astype(np.float32).reshape([-1, 271])[:, :-24]
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32).reshape([1, -1, 1])
    dis_feats = np.repeat(dis_feats, 24, axis=0).reshape([-1, 1])
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1) # , dis_feats
    test_x = feats
    test_y = od_flow.reshape([-1])
    print("test data: ", test_x.shape, test_y.shape)

    # model
    rf_path = logger.model_directory + "rf_noT_" + config["city"] + "_" + str(config["random_seed"]) + ".joblib"
    if os.path.exists(rf_path):
        print("Load model...")
        randomforest = joblib.load(rf_path)
    else:
        print("Init model...")
        randomforest = RandomForestRegressor(n_estimators = 20,
                                            oob_score = True,
                                            max_depth = None,
                                            min_samples_split = 10,
                                            min_samples_leaf = 3,
                                            #   verbose=3,
                                            n_jobs = cpu_count() // 2)
        print("model: ", randomforest)

        print("Fit model...")
        start = time.time()
        randomforest.fit(X = train_x, y = train_y)
        print("Save model...")
        joblib.dump(randomforest, rf_path)
        print('Consume ', time.time()-start, ' seconds!')

    print("Predicting...")
    prediction = randomforest.predict(test_x)

    pre = reMinMax(prediction, input["OD_minmax"])
    od_flow = reMinMax(test_y, input["OD_minmax"])
    rmse = float(RMSE_np(pre, od_flow))
    nrmse = float(NRMSE_np(pre, od_flow))
    mae = float(MAE_np(pre, od_flow))
    mape = float(MAPE_np(pre, od_flow))
    smape = float(SMAPE_np(pre, od_flow))
    cpc = float(CPC_np(pre, od_flow))
    print("RMSE = ", rmse)
    print("NRMSE = ", nrmse)
    print("mae = ", mae)
    print("mape = ", mape)
    print("smape = ", smape)
    print("cpc=", cpc)

def gravity_model(config, input, trainDataloader, validDataloader, testDataloader):
    print("---------- gravity model ----------")
    train_x = []
    train_y = []
    for od_pair_idx, od_flow in trainDataloader:
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.numpy().astype(np.float32)
        o_feats = input["region_node_feats"][od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, 0].reshape([-1, 1])
        d_feats = input["region_node_feats"][od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, 12].reshape([-1, 1])
        dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
        feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
        train_x.append(feats)
        train_y.append(od_flow)
    train_x_lastbatch = train_x[-1]
    train_y_lastbatch = train_y[-1]
    train_x = np.stack(train_x[:-1]).reshape([-1, 3])
    train_y = np.stack(train_y[:-1]).reshape([-1, 24])
    train_x = np.concatenate((train_x, train_x_lastbatch), axis=0)
    train_y = np.concatenate((train_y, train_y_lastbatch), axis=0)
    train_x = torch.FloatTensor(train_x).to(config["device"])
    train_y = torch.FloatTensor(train_y).to(config["device"])
    print("train data: ", train_x.shape, train_y.shape)
    
    # valid data
    od_pair_idx, od_flow = next(iter(validDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, 0].reshape([-1, 1])
    d_feats = input["region_node_feats"][od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, 12].reshape([-1, 1])
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    valid_x = feats
    valid_y = od_flow
    valid_x = torch.FloatTensor(valid_x).to(config["device"])
    valid_y = torch.FloatTensor(valid_y).to(config["device"])
    print("valid data: ", valid_x.shape, valid_y.shape)

    # test data
    od_pair_idx, od_flow = next(iter(testDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, 0].reshape([-1, 1])
    d_feats = input["region_node_feats"][od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, 12].reshape([-1, 1])
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    test_x = feats
    test_y = od_flow
    test_x = torch.FloatTensor(test_x).to(config["device"])
    test_y = torch.FloatTensor(test_y).to(config["device"])
    print("test data: ", test_x.shape, test_y.shape)

    # OD reminmax
    train_y = reMinMax(train_y, input["OD_minmax"])
    valid_y = reMinMax(valid_y, input["OD_minmax"])
    test_y = reMinMax(test_y, input["OD_minmax"])

    # model
    model = single_gravity(config)
    model.to(config["device"])

    # optm
    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=3e-4)

    # train
    for i in tqdm(range(200000)):
        optm.zero_grad()

        pres = model(train_x)

        loss = criterion(pres, train_y)
        # print("Epoch No. " + str(i) + " | loss = ", str(np.sqrt(float(loss.item())))[:12], end=" | valid rmse = ")

        loss.backward()
        optm.step()

        # valid
        with torch.no_grad():
            pres = model(valid_x)
            rmse = RMSE(pres, valid_y)
            # print(str(float(rmse))[:12])

    # test
    with torch.no_grad():
        pres = model(test_x)
        pre = pres.cpu().numpy()
        od_flow = test_y.cpu().numpy()
        rmse = float(RMSE_np(pre, od_flow))
        nrmse = float(NRMSE_np(pre, od_flow))
        mae = float(MAE_np(pre, od_flow))
        mape = float(MAPE_np(pre, od_flow))
        smape = float(SMAPE_np(pre, od_flow))
        cpc = float(CPC_np(pre, od_flow))
        print("RMSE = ", rmse)
        print("NRMSE = ", nrmse)
        print("mae = ", mae)
        print("mape = ", mape)
        print("smape = ", smape)
        print("cpc=", cpc)

def multi_GNN(config, input, trainDataloader, validDataloader, testDataloader):
    print("----- multi-gnn -----")
    # model
    model = multi_gnn(config)
    model.to(config["device"])

    # train
    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=3e-4)
    for epoch in tqdm(range(20000)):
        loss_epoch = []
        for od_pair_idx, od_flow in trainDataloader:
            optm.zero_grad()
            od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
            od_flow = od_flow.float().to(config["device"])
            pre = model(input["region_graph"],
                        input["region_node_feats"],
                        input["distance"],
                        od_pair_idx)
            loss = criterion(od_flow, pre)
            loss_epoch.append(loss.item())
            loss.backward()
            optm.step()
        loss_epoch = float(np.sqrt(np.mean(loss_epoch)))
        # print(" Epoch No.", epoch, " | loss = ", loss_epoch, end="\r")
    
    # test
    with torch.no_grad():
        od_pair_idx, od_flow = next(iter(testDataloader))
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.float().to(config["device"])
        pre = model(input["region_graph"],
                    input["region_node_feats"],
                    input["distance"],
                    od_pair_idx)
        pre = reMinMax(pre, input["OD_minmax"])
        od_flow = reMinMax(od_flow, input["OD_minmax"])
        rmse = float(RMSE(pre, od_flow))
        nrmse = float(NRMSE(pre, od_flow))
        mae = float(MAE(pre, od_flow))
        mape = float(MAPE(pre, od_flow))
        smape = float(SMAPE(pre, od_flow))
        cpc = float(CPC(pre, od_flow))
        print("RMSE = ", rmse)
        print("NRMSE = ", nrmse)
        print("mae = ", mae)
        print("mape = ", mape)
        print("smape = ", smape)
        print("cpc=", cpc)

def one_GNN(config, input, trainDataloader, validDataloader, testDataloader):
    print("----- one-gnn -----")
    # model
    model = one_gnn(config)
    model.to(config["device"])

    # train
    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=3e-4)
    for epoch in tqdm(range(20000)):
        loss_epoch = []
        for od_pair_idx, od_flow in trainDataloader:
            optm.zero_grad()
            od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
            od_flow = od_flow.float().to(config["device"])
            pre = model(input["region_graph"],
                        input["region_node_feats"],
                        input["distance"],
                        od_pair_idx)
            loss = criterion(od_flow, pre)
            loss_epoch.append(loss.item())
            loss.backward()
            optm.step()
        loss_epoch = float(np.sqrt(np.mean(loss_epoch)))

    # test
    with torch.no_grad():
        od_pair_idx, od_flow = next(iter(testDataloader))
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.float().to(config["device"])
        pre = model(input["region_graph"],
                    input["region_node_feats"],
                    input["distance"],
                    od_pair_idx)
        pre = reMinMax(pre, input["OD_minmax"])
        od_flow = reMinMax(od_flow, input["OD_minmax"])
        rmse = float(RMSE(pre, od_flow))
        nrmse = float(NRMSE(pre, od_flow))
        mae = float(MAE(pre, od_flow))
        mape = float(MAPE(pre, od_flow))
        smape = float(SMAPE(pre, od_flow))
        cpc = float(CPC(pre, od_flow))
        print("RMSE = ", rmse)
        print("NRMSE = ", nrmse)
        print("mae = ", mae)
        print("mape = ", mape)
        print("smape = ", smape)
        print("cpc=", cpc)

def multi_gravity_model(config, input, trainDataloader, validDataloader, testDataloader, logger):
    print("---------- multi gravity model ----------")
    train_x = []
    train_y = []
    for od_pair_idx, od_flow in trainDataloader:
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.numpy().astype(np.float32)
        o_feats = input["region_node_feats"][od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, 0].reshape([-1, 1])
        d_feats = input["region_node_feats"][od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, 12].reshape([-1, 1])
        dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
        feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
        train_x.append(feats)
        train_y.append(od_flow)
    train_x_lastbatch = train_x[-1]
    train_y_lastbatch = train_y[-1]
    train_x = np.stack(train_x[:-1]).reshape([-1, 3])
    train_y = np.stack(train_y[:-1]).reshape([-1, 24])
    train_x = np.concatenate((train_x, train_x_lastbatch), axis=0)
    train_y = np.concatenate((train_y, train_y_lastbatch), axis=0)
    train_x = torch.FloatTensor(train_x).to(config["device"])
    train_y = torch.FloatTensor(train_y).to(config["device"])
    print("train data: ", train_x.shape, train_y.shape)
    
    # valid data
    od_pair_idx, od_flow = next(iter(validDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, 0].reshape([-1, 1])
    d_feats = input["region_node_feats"][od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, 12].reshape([-1, 1])
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    valid_x = feats
    valid_y = od_flow
    valid_x = torch.FloatTensor(valid_x).to(config["device"])
    valid_y = torch.FloatTensor(valid_y).to(config["device"])
    print("valid data: ", valid_x.shape, valid_y.shape)

    # test data
    od_pair_idx, od_flow = next(iter(testDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, 0].reshape([-1, 1])
    d_feats = input["region_node_feats"][od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, 12].reshape([-1, 1])
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    test_x = feats
    test_y = od_flow
    test_x = torch.FloatTensor(test_x).to(config["device"])
    test_y = torch.FloatTensor(test_y).to(config["device"])
    print("test data: ", test_x.shape, test_y.shape)

    # OD reminmax
    train_y = reMinMax(train_y, input["OD_minmax"])
    valid_y = reMinMax(valid_y, input["OD_minmax"])
    test_y = reMinMax(test_y, input["OD_minmax"])

    # model
    model = multi_gravity(config)
    model.to(config["device"])

    # optm
    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=3e-4)

    # train
    for i in tqdm(range(200000)):
        optm.zero_grad()

        pres = model(train_x)

        loss = criterion(pres, train_y)
        # print("Epoch No. " + str(i) + " | loss = ", str(np.sqrt(float(loss.item())))[:12], end=" | valid rmse = ")

        loss.backward()
        optm.step()

        # valid
        with torch.no_grad():
            pres = model(valid_x)
            rmse = RMSE(pres, valid_y)
            # print(str(float(rmse))[:12])

    # test
    with torch.no_grad():
        pres = model(test_x)
        pre = pres.cpu().numpy()
        od_flow = test_y.cpu().numpy()
        rmse = float(RMSE_np(pre, od_flow))
        nrmse = float(NRMSE_np(pre, od_flow))
        mae = float(MAE_np(pre, od_flow))
        mape = float(MAPE_np(pre, od_flow))
        smape = float(SMAPE_np(pre, od_flow))
        cpc = float(CPC_np(pre, od_flow))
        print("RMSE = ", rmse)
        print("NRMSE = ", nrmse)
        print("mae = ", mae)
        print("mape = ", mape)
        print("smape = ", smape)
        print("cpc=", cpc)

def one_deepgravity(config, input, trainDataloader, validDataloader, testDataloader):
    print("---------- single deep gravity model ----------")
    train_x = []
    train_y = []
    for od_pair_idx, od_flow in trainDataloader:
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.numpy().astype(np.float32)
        o_feats = input["region_node_feats"][0, od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, :-24]
        d_feats = input["region_node_feats"][0, od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, :-24]
        dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
        feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
        train_x.append(feats)
        train_y.append(od_flow)
    train_x_lastbatch = train_x[-1]
    train_y_lastbatch = train_y[-1]
    train_x = np.stack(train_x[:-1]).reshape([-1, 495])
    train_y = np.stack(train_y[:-1]).reshape([-1, 24])
    train_x = np.concatenate((train_x, train_x_lastbatch), axis=0)
    train_y = np.concatenate((train_y, train_y_lastbatch), axis=0)
    train_x = torch.FloatTensor(train_x).to(config["device"])
    train_y = torch.FloatTensor(train_y).to(config["device"])
    print("train data: ", train_x.shape, train_y.shape)
    
    # valid data
    od_pair_idx, od_flow = next(iter(validDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][0, od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, :-24]
    d_feats = input["region_node_feats"][0, od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, :-24]
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    valid_x = feats
    valid_y = od_flow
    valid_x = torch.FloatTensor(valid_x).to(config["device"])
    valid_y = torch.FloatTensor(valid_y).to(config["device"])
    print("valid data: ", valid_x.shape, valid_y.shape)

    # test data
    od_pair_idx, od_flow = next(iter(testDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][0, od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, :-24]
    d_feats = input["region_node_feats"][0, od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, :-24]
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    test_x = feats
    test_y = od_flow
    test_x = torch.FloatTensor(test_x).to(config["device"])
    test_y = torch.FloatTensor(test_y).to(config["device"])
    
    # model
    model = one_DG(config)
    model.to(config["device"])

    # optm
    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=1e-4)

    # train
    for i in tqdm(range(7000)):
        optm.zero_grad()
        pres = model(train_x)
        loss = criterion(pres, train_y)
        # print("Epoch No. " + str(i) + " | loss = ", str(float(loss.item()))[:12])
        loss.backward()
        optm.step()

    # test
    with torch.no_grad():
        pres = model(test_x)
        test_y = reMinMax(test_y, input["OD_minmax"])
        pres = reMinMax(pres, input["OD_minmax"])
        pre = pres.cpu().numpy()
        od_flow = test_y.cpu().numpy()
        rmse = float(RMSE_np(pre, od_flow))
        nrmse = float(NRMSE_np(pre, od_flow))
        mae = float(MAE_np(pre, od_flow))
        mape = float(MAPE_np(pre, od_flow))
        smape = float(SMAPE_np(pre, od_flow))
        cpc = float(CPC_np(pre, od_flow))
        print("RMSE = ", rmse)
        print("NRMSE = ", nrmse)
        print("mae = ", mae)
        print("mape = ", mape)
        print("smape = ", smape)
        print("cpc=", cpc)

def multi_deepgravity(config, input, trainDataloader, validDataloader, testDataloader):
    print("---------- multi deep gravity model ----------")
    train_x = []
    train_y = []
    for od_pair_idx, od_flow in trainDataloader:
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.numpy().astype(np.float32)
        o_feats = input["region_node_feats"][0, od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, :-24]
        d_feats = input["region_node_feats"][0, od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, :-24]
        dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
        feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
        train_x.append(feats)
        train_y.append(od_flow)
    train_x_lastbatch = train_x[-1]
    train_y_lastbatch = train_y[-1]
    train_x = np.stack(train_x[:-1]).reshape([-1, 495])
    train_y = np.stack(train_y[:-1]).reshape([-1, 24])
    train_x = np.concatenate((train_x, train_x_lastbatch), axis=0)
    train_y = np.concatenate((train_y, train_y_lastbatch), axis=0)
    train_x = torch.FloatTensor(train_x).to(config["device"])
    train_y = torch.FloatTensor(train_y).to(config["device"])
    print("train data: ", train_x.shape, train_y.shape)
    
    # valid data
    od_pair_idx, od_flow = next(iter(validDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][0, od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, :-24]
    d_feats = input["region_node_feats"][0, od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, :-24]
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    valid_x = feats
    valid_y = od_flow
    valid_x = torch.FloatTensor(valid_x).to(config["device"])
    valid_y = torch.FloatTensor(valid_y).to(config["device"])
    print("valid data: ", valid_x.shape, valid_y.shape)

    # test data
    od_pair_idx, od_flow = next(iter(testDataloader))
    od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
    od_flow = od_flow.numpy().astype(np.float32)
    o_feats = input["region_node_feats"][0, od_pair_idx[0], :].cpu().numpy().astype(np.float32)[:, :-24]
    d_feats = input["region_node_feats"][0, od_pair_idx[1], :].cpu().numpy().astype(np.float32)[:, :-24]
    dis_feats = input["distance"][od_pair_idx].reshape([-1, 1]).cpu().numpy().astype(np.float32)
    feats = np.concatenate((o_feats, d_feats, dis_feats), axis=1)
    test_x = feats
    test_y = od_flow
    test_x = torch.FloatTensor(test_x).to(config["device"])
    test_y = torch.FloatTensor(test_y).to(config["device"])
    
    # model
    model = multi_DG(config)
    model.to(config["device"])

    # optm
    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=1e-4)

    # train
    for i in tqdm(range(7000)):
        optm.zero_grad()
        pres = model(train_x)
        loss = criterion(pres, train_y)
        # print("Epoch No. " + str(i) + " | loss = ", str(float(loss.item()))[:12])
        loss.backward()
        optm.step()

    # test
    with torch.no_grad():
        pres = model(test_x)
        test_y = reMinMax(test_y, input["OD_minmax"])
        pres = reMinMax(pres, input["OD_minmax"])
        pre = pres.cpu().numpy()
        od_flow = test_y.cpu().numpy()
        rmse = float(RMSE_np(pre, od_flow))
        nrmse = float(NRMSE_np(pre, od_flow))
        mae = float(MAE_np(pre, od_flow))
        mape = float(MAPE_np(pre, od_flow))
        smape = float(SMAPE_np(pre, od_flow))
        cpc = float(CPC_np(pre, od_flow))
        print("RMSE = ", rmse)
        print("NRMSE = ", nrmse)
        print("mae = ", mae)
        print("mape = ", mape)
        print("smape = ", smape)
        print("cpc=", cpc)
    

if __name__ == "__main__":
    print(cpu_count())