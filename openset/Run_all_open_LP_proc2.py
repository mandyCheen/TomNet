import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import torch

warnings.filterwarnings("ignore")

settings = [("5way_5shot", 10, 1.0), ("5way_10shot", 11, 1.0), ("10way_5shot", 11, 1.0), ("5way_5shot", 22, 1.0), ("10way_10shot", 22, 1.0)]

for exp, seed, lambda_ in settings:
    options = load_config("../config/config_label_prop_openset_meta_nict_proc2.json")
    print("exp: ", exp)
    print("lambda: ", lambda_)
    options["settings"]["name"] = exp+f"_LabelPropagation_alpha0.7_k20_gcn_lambda{lambda_}"
    shots = int(exp.split("_")[1].split("shot")[0])
    way = int(exp.split("_")[0].split("way")[0])
    options["settings"]["few_shot"]["train"]["support_shots"] = shots
    options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
    options["settings"]["few_shot"]["test"]["support_shots"] = shots
    options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
    options["settings"]["few_shot"]["train"]["class_per_iter"] = way
    options["settings"]["few_shot"]["test"]["class_per_iter"] = way
    options["settings"]["train"]["device"] = "cuda:2"
    options["settings"]["openset"]["train"]["loss_weight"] = lambda_
    options["settings"]["seed"] = seed
    save_config(options, "../config/config_label_prop_openset_meta_nict_proc2.json")

    dataset = LoadDataset(options)

    trainModule = TrainModule(options, dataset)
    trainModule.train()

    # test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
    # test.eval()

    torch.cuda.empty_cache()