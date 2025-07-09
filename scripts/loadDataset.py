
import pandas as pd
import numpy as np
import os

class LoadDataset:
    def __init__(self, opt: dict):
            rawDatasetPath = os.path.join(opt["paths"]["data"]["csv_folder"], opt["dataset"]["raw"])
            self.rawDataset = pd.read_csv(rawDatasetPath)
            self.seed = opt["settings"]["seed"]
            self.cpuArch = opt["dataset"]["cpu_arch"]
            self.datasetSplitFolder = opt["paths"]["data"]["split_folder"]
            self.val = opt["settings"]["train"]["validation"]
            self.reverseTool = opt["dataset"]["reverse_tool"]
            val = "_withVal" if self.val else ""
            self.datasetName = f"{self.cpuArch}{val}_{self.reverseTool}_{self.seed}"
            self.trainData, self.testData, self.valData = self.load_all_datasets()
            ## openset
            self.enable_openset = opt.get("dataset", {}).get("openset", False)
            if self.enable_openset:
                rawOSDatasetPath = os.path.join(opt["paths"]["data"]["csv_folder"], opt["dataset"]["openset_raw"])
                rawOSDataset = pd.read_csv(rawOSDatasetPath)
                self.opensetDataRatio = opt["dataset"]["openset_data_ratio"]
                self.opensetData = self.load_openset_data(rawOSDataset=rawOSDataset, mode=opt["dataset"]["openset_data_mode"])
            else:
                self.opensetData = None
    
    def write_split_dataset(self, mode, familyList) -> None:
        if not os.path.exists(self.datasetSplitFolder):
            os.makedirs(self.datasetSplitFolder)
        filepath = f"{self.datasetSplitFolder}/{mode}_{self.datasetName}.txt"
        with open(filepath, "w") as f:
            for family in familyList:
                f.write(f"{family}\n")

    def get_split_dataset(self) -> None: 
        if self.val:
            testNum, valNum = 10, 10
        else:
            testNum, valNum = 10, 0
        allFamilies = set(self.rawDataset["family"].unique())

        allFamiliesList = list(allFamilies)

        np.random.seed(self.seed)
        np.random.shuffle(allFamiliesList)
        
        TrainNum = len(allFamilies) - testNum - valNum

        trainFamily = allFamiliesList[:TrainNum]
        testFamily = allFamiliesList[TrainNum:TrainNum + testNum]
        valFamily = allFamiliesList[TrainNum + testNum:]
        
        self.write_split_dataset("train", trainFamily)
        self.write_split_dataset("test", testFamily)
        if self.val:
            self.write_split_dataset("val", valFamily)
    
    def load_dataset(self, mode) -> pd.DataFrame:
        filepath = f"{self.datasetSplitFolder}/{mode}_{self.datasetName}.txt"
    
        if not os.path.exists(filepath):
            print(f"Split dataset for {mode} does not exist, creating split dataset...")
            self.get_split_dataset()
    
        with open(filepath, "r") as f:
            familyList = f.read().splitlines()
        
        data = self.rawDataset[self.rawDataset["family"].isin(familyList)]
        data = data.reset_index(drop=True)
        print(f"{mode} dataset shape: {data.shape}")
        print(f"{mode} dataset family number: {len(data['family'].unique())}")
        return data

    def load_all_datasets(self) -> pd.DataFrame:
        print("Loading all datasets...")
        trainData = self.load_dataset("train")
        testData = self.load_dataset("test")
        valData = self.load_dataset("val") if self.val else None
        return trainData, testData, valData

    def load_openset_data(self, rawOSDataset: pd.DataFrame, mode: str) -> pd.DataFrame:
        print("Loading openset data...")

        if mode == "all":
            opensetData = rawOSDataset
        elif mode == "random":
            opensetData = rawOSDataset.sample(frac=self.opensetDataRatio, random_state=self.seed)
            opensetData = opensetData.reset_index(drop=True)

        print(f"Openset data shape: {opensetData.shape}")
        return opensetData
