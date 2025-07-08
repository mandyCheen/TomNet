import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LoadDatasetPretrained:
    def __init__(self, opt: dict):
        rawDatasetPath = os.path.join(opt["paths"]["data"]["csv_folder"], opt["pretrain"]["raw_dataset"])
        self.rawDataset = pd.read_csv(rawDatasetPath)
        self.seed = opt["settings"]["seed"]
        self.cpuArch = opt["dataset"]["cpu_arch"]
        self.reverseTool = opt["dataset"]["reverse_tool"]
        self.datasetName = f"{self.cpuArch}_pretrain_{self.reverseTool}"
        self.trainData, self.testData, self.valData = self.load_pretrain_dataset()    
    
    def load_pretrain_dataset(self, splitRate: tuple = (0.6, 0.2, 0.2)) -> pd.DataFrame:
        print("Loading pretrain dataset...")
        trainData, testData = train_test_split(self.rawDataset, test_size=splitRate[1], random_state=self.seed)
        trainData, valData = train_test_split(trainData, test_size=splitRate[2], random_state=self.seed)
        return trainData, testData, valData