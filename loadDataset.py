
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class LoadDataset:
    def __init__(self, opt: dict, pretrain: bool = False):
        if pretrain:
            rawDatasetPath = os.path.join(opt["paths"]["data"]["csv_folder"], opt["pretrain"]["raw_dataset"])
            self.rawDataset = pd.read_csv(rawDatasetPath)
            self.seed = opt["settings"]["seed"]
            self.cpuArch = opt["dataset"]["cpu_arch"]
            self.reverseTool = opt["dataset"]["reverse_tool"]
            self.datasetName = f"{self.cpuArch}_pretrain_{self.reverseTool}"
            self.trainData, self.testData, self.valData = self.load_pretrain_dataset()    
        else:
            rawDatasetPath = os.path.join(opt["paths"]["data"]["csv_folder"], opt["dataset"]["raw"])
            self.rawDataset = pd.read_csv(rawDatasetPath)
            self.seed = opt["settings"]["seed"]
            self.cpuArch = opt["dataset"]["cpu_arch"]
            self.datasetSplitFolder = opt["paths"]["data"]["split_folder"]
            self.val = opt["settings"]["train"]["validation"]
            self.splitByCpu = opt["dataset"]["split_by_cpu"]
            self.reverseTool = opt["dataset"]["reverse_tool"]
            self.familyInTrain = opt.get("dataset", {}).get("pretrain_family", [])
            splitByCpu = "_splitByCpu" if self.splitByCpu else ""
            val = "_withVal" if self.val else ""
            pretrain_ = "_withPretrain" if opt["pretrain"]["use"] else ""
            self.familyCpuList = self.rawDataset.groupby("family")["CPU"].unique().to_dict()
            self.datasetName = f"{self.cpuArch}{splitByCpu}{val}{pretrain_}_{self.reverseTool}_{self.seed}"
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
        requiredTrainFamilies = set(self.familyInTrain)

        missing_families = requiredTrainFamilies - allFamilies
        if missing_families:
            raise ValueError(f"Required families not found in dataset: {missing_families}")

        remainingFamilies = list(allFamilies - requiredTrainFamilies)

        np.random.seed(self.seed)
        np.random.shuffle(remainingFamilies)
        
        additionalTrainNum = len(allFamilies) - testNum - valNum - len(requiredTrainFamilies)

        trainFamily = list(requiredTrainFamilies) + remainingFamilies[:additionalTrainNum]
        testFamily = remainingFamilies[additionalTrainNum:additionalTrainNum + testNum]
        valFamily = remainingFamilies[additionalTrainNum + testNum:]
        
        self.write_split_dataset("train", trainFamily)
        self.write_split_dataset("test", testFamily)
        if self.val:
            self.write_split_dataset("val", valFamily)
    
    def get_split_dataset_by_cpu(self) -> None:
        if self.val:
            testRate, valRate = 0.75, 0.25
        else:
            testRate, valRate = 1, 0.0
        cpuCounts = self.rawDataset["CPU"].value_counts()
        print(f"CPU counts: {cpuCounts}")

        # train cpu is the cpu with the most samples
        trainCpu = cpuCounts.idxmax()
        testCpu = cpuCounts.index[cpuCounts.index != trainCpu][0]

        trainFamilyList = self.rawDataset[self.rawDataset["CPU"] == trainCpu]["family"].unique()
        testFamilyList = self.rawDataset[self.rawDataset["CPU"] == testCpu]["family"].unique()

        np.random.seed(self.seed)
        np.random.shuffle(testFamilyList)

        trainFamily = trainFamilyList
        testFamily = testFamilyList[:int(len(testFamilyList) * testRate)]
        valFamily = testFamilyList[int(len(testFamilyList) * testRate):]
        self.write_split_dataset("train", trainFamily)
        self.write_split_dataset("test", testFamily)
        if self.val:
            self.write_split_dataset("val", valFamily)

    def load_dataset(self, mode) -> pd.DataFrame:
        filepath = f"{self.datasetSplitFolder}/{mode}_{self.datasetName}.txt"
        if self.splitByCpu:
            if not os.path.exists(filepath):
                print(f"Split dataset for {mode} does not exist, creating split dataset...")
                self.get_split_dataset_by_cpu()        
        else:
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
    
    def load_pretrain_dataset(self, splitRate: tuple = (0.6, 0.2, 0.2)) -> pd.DataFrame:
        print("Loading pretrain dataset...")
        trainData, testData = train_test_split(self.rawDataset, test_size=splitRate[1], random_state=self.seed)
        trainData, valData = train_test_split(trainData, test_size=splitRate[2], random_state=self.seed)
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
