import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule
import torch
import datetime

warnings.filterwarnings("ignore")

expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
seeds = [6, 22, 31, 42, 7, 10, 666, 11, 19]
lambdaList = [1.5, 2.0]

for seed in seeds:
    print("seed: ", seed)
    for exp in expList:
        for lambda_ in lambdaList:
            if seed == 6:
                if exp == "5way_5shot" or exp == "5way_10shot":
                    continue
                if exp == "10way_5shot" and lambda_ == 1.5:
                    continue
            options = load_config("../config/config_label_prop_openset_meta_nict_proc1.json")
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
            options["settings"]["train"]["device"] = "cuda:1"
            options["settings"]["openset"]["train"]["loss_weight"] = lambda_
            options["settings"]["seed"] = seed
            save_config(options, "../config/config_label_prop_openset_meta_nict_proc1.json")

            dataset = LoadDataset(options)

            try:
                trainModule = TrainModule(options, dataset)
                trainModule.train()
            except Exception as e:
                now = datetime.datetime.now()
                open(f"./logs/Error_log_{now.strftime('%Y%m%d_%H%M')}.txt", "a").write(f"{now}: {exp} seed:{seed} lambda:{lambda_} {e}\n")
            # test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
            # test.eval()

            torch.cuda.empty_cache()