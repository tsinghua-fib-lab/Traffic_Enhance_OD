import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint

from utils.metrics import *


def evaluation(config, input, model, testDataloader, logger):
    print("\n**********  evaluation  **********")
    model.load_state_dict(torch.load(logger.model_path, map_location=torch.device(config["device"])))
    model.to(config["device"])
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
        logger.log_results(rmse=rmse, nrmse=nrmse, mae=mae, 
                           mape=mape, smape=smape, cpc=cpc)
        logger.save_exp_log()
        logger.log_test_prediction(pre)
        logger.log_test_groundtruth(od_flow)

    print("RMSE = ", rmse)
    print("NRMSE = ", nrmse)
    print("mae = ", mae)
    print("mape = ", mape)
    print("smape = ", smape)
    print("cpc=", cpc)

    print("**********************************")