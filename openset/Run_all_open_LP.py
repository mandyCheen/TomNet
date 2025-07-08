import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import torch

warnings.filterwarnings("ignore")

expList = ["5way_10shot", "10way_5shot", "10way_10shot"]
# seeds = [6, 22, 31, 42, 888, 7, 10, 666, 11, 19]
# lambdaList = [1.0, 1.5, 2.0]

for exp in expList:
    options = load_config("../config/config_label_prop_openset_meta_14.json")
    print("exp: ", exp)
    options["settings"]["name"] = exp+"_LabelPropagation_alpha0.7_k20_gcn"+f"_lambda1.0"
    shots = int(exp.split("_")[1].split("shot")[0])
    way = int(exp.split("_")[0].split("way")[0])
    options["settings"]["few_shot"]["train"]["support_shots"] = shots
    options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
    options["settings"]["few_shot"]["test"]["support_shots"] = shots
    options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
    options["settings"]["few_shot"]["train"]["class_per_iter"] = way
    options["settings"]["few_shot"]["test"]["class_per_iter"] = way

    save_config(options, "../config/config_label_prop_openset_meta_14.json")

    dataset = LoadDataset(options)

    trainModule = TrainModule(options, dataset)
    trainModule.train()

    test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
    test.eval()

torch.cuda.empty_cache()