from eviltransform import distance
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from utils.metrics import *

def train(config, input, model, trainDataloader, validDataloader, logger):
    region_graph = input["region_graph"]
    bigraph = input["bigraph"]
    region_node_feats = input["region_node_feats"]
    distance = input["distance"]
    
    model.train()

    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=config["lr"])
    if config["mode"] != "init":
        optm.load_state_dict(torch.load(logger.optimizer_path))

    # iteration
    print("**********  training  **********")
    for epoch in range(config["EPOCH"]):
        loss_epoch = []
        for od_pair_idx, od_flow in trainDataloader:
            optm.zero_grad()
            od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])

            od_flow = od_flow.float().to(config["device"])
            pre = model(region_graph,
                        bigraph,
                        region_node_feats, 
                        distance,
                        od_pair_idx)
            loss = criterion(od_flow, pre)
            loss_epoch.append(loss.item())
            loss.backward()
            optm.step()

        print("Epoch No.", epoch)
        loss_epoch = float(np.sqrt(np.mean(loss_epoch)))
        logger.writer.add_scalar("Train/loss", loss_epoch, epoch)
        logger.log_training_loss(loss_epoch)
        print("loss=" + str(loss_epoch)[:8], end=" | valid RMSE=")
        

        # valid
        with torch.no_grad():
            od_pair_idx, od_flow = next(iter(validDataloader))
            od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
            od_flow = od_flow.float().to(config["device"])
            pre = model(region_graph,
                        region_node_feats,
                        distance,
                        od_pair_idx)
            pre = reMinMax(pre, input["OD_minmax"])
            od_flow = reMinMax(od_flow, input["OD_minmax"])
            rmse = float(RMSE(pre, od_flow))
            smape = float(SMAPE(pre, od_flow))
            cpc = float(CPC(pre, od_flow))
            print(str(rmse)[:8] + " | SMAPE="+str(smape)[:8] + " | CPC=" + str(cpc)[:8])
            logger.writer.add_scalar("valid/rmse", rmse, epoch)
            logger.writer.add_scalar("valid/smape", smape, epoch)
            logger.writer.add_scalar("valid/cpc", cpc, epoch)
            logger.check_save_model(rmse, model, optm)

        if logger.check_converge() or logger.check_overfitting(rmse):
            break

        logger.save_exp_log()

    return model
