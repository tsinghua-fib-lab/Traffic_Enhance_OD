import os
import json
import datetime

from train import *
from evaluation import *
from utils.log import *
from utils.procedure import *
from utils.data_loader import *
from utils.data_preprocessing import *
from models.ours import ST_transor, combined
from models.baselines import *

from torch.utils.data import DataLoader

from setproctitle import setproctitle



def main(config):
    # preprocessing
    data_np = preprocess(config)

    # prepare dataloader
    data = Urban_dataset(config)
    data.getData(data_np)
    dataset = IDX_datasets(config)
    dataset.getOD(data.OD)
    trainDataset, validDataset, testDataset = data_split(dataset, config)
    trainDataloader = DataLoader(dataset=trainDataset, batch_size=config["batch_size"], shuffle=True)
    validDataloader = DataLoader(dataset=validDataset, batch_size=len(validDataset))
    testDataloader = DataLoader(dataset=testDataset, batch_size=len(testDataset))
    print("******  prepared dataloader for pytorch  *******\n")

    # features
    print("******  preparing tensor for pytorch  *******")
    region_node_feats = torch.FloatTensor(data.node_feats).to(config["device"])
    distance = torch.FloatTensor(data.distance).to(config["device"])
    region_graph = build_DGLGraph(data.adjacency).to(config["device"])
    input = {}
    input["OD_minmax"] = data.OD_minmax
    input["region_node_feats"] = region_node_feats
    input["distance"] = distance
    input["region_graph"] = region_graph
    input["bigraph"] = data.traffic_graph
    config["GNN_in_dim"] = region_node_feats.size(-1) - 2 + config["node_embsize"]
    print(distance.size())
    print(region_node_feats.size())
    print(region_graph)
    print("*"*len("******  prepared tensor for pytorch  *******")+"\n")

    # logger
    logger = Logger(config)
    # print("test logger:")
    # cmd = "ls -l " + logger.groundtruth_path
    # print(cmd)
    # os.system(cmd)
    # exit()


    # baselines
    if config["baseline"] != 0:
        print("**************** baselines ****************")
        config["GNN_in_dim"] = region_node_feats.size(-1) - 24
        if config["baseline"] == "multi-GNN":
            multi_GNN(config, input, trainDataloader, validDataloader, testDataloader)
        elif config["baseline"] == "one-GNN":
            config["GNN_in_dim"] = region_node_feats.size(-1)
            one_GNN(config, input, trainDataloader, validDataloader, testDataloader)
        elif config["baseline"] == "rf":
            random_forest(config, input, trainDataloader, validDataloader, testDataloader, logger)
        elif config["baseline"] == "rf-noT":
            random_forest_noT(config, input, trainDataloader, validDataloader, testDataloader, logger)
        elif config["baseline"] == "multi-rf":
            multi_rf(config, input, trainDataloader, validDataloader, testDataloader, logger)
        elif config["baseline"] == "multi-GM":
            input["region_node_feats"] = torch.FloatTensor(data.region_attributes_raw).to(config["device"])
            input["distance"] = torch.FloatTensor(data.distance_raw).to(config["device"])
            multi_gravity_model(config, input, trainDataloader, validDataloader, testDataloader, logger)
        elif config["baseline"] == "GM":
            input["region_node_feats"] = torch.FloatTensor(data.region_attributes_raw).to(config["device"])
            input["distance"] = torch.FloatTensor(data.distance_raw).to(config["device"])
            gravity_model(config, input, trainDataloader, validDataloader, testDataloader)
        elif config["baseline"] == "DG":
            one_deepgravity(config, input, trainDataloader, validDataloader, testDataloader)
        elif config["baseline"] == "multi-DG":
            multi_deepgravity(config, input, trainDataloader, validDataloader, testDataloader)
        else:
            raise Exception("Unknown baselines: " + config["baseline"])
        print(len("**************** baselines ****************") * "*")
        exit()


    # model
    model = combined(config).to(config["device"])
    if config["mode"] == "init":
        print("********** init model **********")
    else:
        print("********** load model **********")
        model.load_state_dict(torch.load(logger.model_path, map_location=torch.device(config["device"])))
        model.to(config["device"])
    print(model)
    print("*"*len("********** init model **********")+"\n")

    # train
    if config["eval"] == "no":
        # with torch.autograd.set_detect_anomaly(True):
        train(config, input, model, trainDataloader, validDataloader, logger)

    # test
    model = model.ODpart
    evaluation(config, input, model, testDataloader, logger)



if __name__ == "__main__":

    # config
    config = get_conifg("src/config/beijing_dl4.json")

    # random seed
    setRandomSeed(config["random_seed"])

    main(config)
