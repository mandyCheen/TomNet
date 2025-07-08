from logging import config
from pyexpat import model
import warnings, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset
from utils import load_config, save_config
from loadDataset import LoadDataset
from trainModule import TestModule
warnings.filterwarnings("ignore")

expList = ["10way_5shot", "10way_10shot"]
seeds = [19]
lambdaList = [1.0, 1.5, 2.0]

for seed in seeds:
    rootFolder = f"../checkpoints/x86_64_withVal_withPretrain_ghidra_{seed}_openset"
    for exp in expList:
        print("exp: ", exp)
        for folder in os.listdir(rootFolder):
            if os.path.isdir(os.path.join(rootFolder, folder)):
                if "lambda" not in folder:
                    continue
                if exp not in folder:
                    continue
                lambda_value = float((folder.split("_")[6]).split("lambda")[1])
                if lambda_value not in lambdaList:
                    continue
                if seed == 19 and lambda_value == 2.0 and exp == "10way_5shot":
                    continue
                print("modelFolder: ", folder)
                modelFolder = folder
                modelPath = os.path.join(rootFolder, modelFolder)

                configPath = os.path.join(modelPath, "config.json")
                options = load_config(configPath)
                options["paths"]["data"]["fcg_dataset"] = "/mnt/ssd2t/mandy/Projects/few_shot_fcg/dataset/data_ghidra_fcg"
                options["paths"]["data"]["embedding_folder"] = "/mnt/ssd2t/mandy/Projects/few_shot_fcg/embeddings"
                options["paths"]["data"]["openset_dataset"] = "/mnt/ssd2t/mandy/Projects/few_shot_fcg/dataset/data_ghidra_fcg_openset"
                options["settings"]["train"]["device"] = "cuda:0"

                save_config(options, configPath)
                dataset = LoadDataset(options)
                test = TestModule(configPath=configPath, dataset=dataset)
                test.eval()