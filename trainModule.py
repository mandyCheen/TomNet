import pandas as pd
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Batch
from loadDataset import LoadDataset
from dataset import FcgSampler, OpenSetFcgSampler
from models import *
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from loss import *
from datetime import datetime
from train_utils import Training, Testing, load_GE_data
from utils import load_config, record_log

def collate_graphs(batch):
    return Batch.from_data_list(batch)

class TrainModule(Training):
    def __init__(self, opt: dict, dataset: LoadDataset):

        self.embeddingFolder = os.path.join(opt["paths"]["data"]["embedding_folder"], dataset.datasetName, opt["settings"]["vectorize"]["node_embedding_method"])
        self.rawDataset = dataset.rawDataset
        self.trainDataset = dataset.trainData
        self.valDataset = dataset.valData
        self.testDataset = dataset.testData
        self.opensetData = dataset.opensetData
        self.trainGraph = []
        self.valGraph = []
        self.testGraph = []
        self.loss_fn = None
        self.opt = opt
        self.datasetName = dataset.datasetName
        self.folderName = dataset.datasetName
        if opt["dataset"]["addition_note"]:
            self.folderName += f"_{opt['dataset']['addition_note']}"

        self.support_shots_train = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.query_shots_train = opt["settings"]["few_shot"]["train"]["query_shots"]
        self.class_per_iter_train = opt["settings"]["few_shot"]["train"]["class_per_iter"]
        self.support_shots_test = opt["settings"]["few_shot"]["test"]["support_shots"]
        self.query_shots_test = opt["settings"]["few_shot"]["test"]["query_shots"]
        self.class_per_iter_test = opt["settings"]["few_shot"]["test"]["class_per_iter"]
        
        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        if self.enable_openset:
            self.openset_m_samples = opt["settings"]["openset"]["train"]["m_samples"]
            self.openset_class_per_iter = opt["settings"]["openset"]["train"]["class_per_iter"]
            self.openset_m_samples_test = opt["settings"]["openset"]["test"]["m_samples"]

        self.iterations = opt["settings"]["train"]["iterations"]
        self.device = opt["settings"]["train"]["device"]
        self.embeddingSize = opt["settings"]["vectorize"]["node_embedding_size"]
        self.lr_scheduler = opt["settings"]["train"]["lr_scheduler"]["use"]

        self.parallel = opt["settings"]["train"]["parallel"]
        self.parallel_device = opt["settings"]["train"]["parallel_device"]

        now = datetime.now()
        self.model_folder = opt["paths"]["model"]["model_folder"] + "/" + self.folderName + "/" + opt["settings"]["name"] + "_" + now.strftime("%Y%m%d_%H%M%S")
        self.pretrain_folder = opt["paths"]["model"]["pretrained_folder"]
        os.makedirs(self.model_folder, exist_ok=True)
        self.log_file = self.model_folder + "/log.txt"
   
        self.setting()

        super().__init__(
            opt=self.opt,
            trainLoader=self.trainLoader,
            valLoader=self.valLoader,
            model=self.model,
            loss_fn=self.loss_fn,
            optim=self.optim,
            scheduler=self.scheduler if self.lr_scheduler else None,
            device=self.device,
            model_path=self.model_folder,
        )

    def get_backbone(self):
        info = self.opt["settings"]["model"]
        if info["model_name"] == "GraphSAGE":
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GraphSAGELayer(dim_in=info["input_size"], dim_h=info["hidden_size"], 
                                            dim_o=info["output_size"], num_layers=info["num_layers"])
            elif self.opt["settings"]["few_shot"]["method"] == "MAML":
                self.model = GraphClassifier(backbone_dim_in=info["input_size"], backbone_dim_h=info["hidden_size"],
                                             backbone_dim_out=info["output_size"], backbone_num_layers=info["num_layers"], 
                                            projection = info["projection"], num_classes=self.opt["settings"]["few_shot"]["train"]["class_per_iter"],
                                            backbone_type='sage')
            else:
                self.model = GraphSAGE(dim_in=info["input_size"], dim_h=info["hidden_size"], 
                                       dim_out=info["output_size"], num_layers=info["num_layers"], 
                                       projection = info["projection"])
        elif info["model_name"] == "GCN":
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GCNLayer(dim_in=info["input_size"], dim_h=info["hidden_size"], 
                                      dim_o=info["output_size"], num_layers=info["num_layers"])
            elif self.opt["settings"]["few_shot"]["method"] == "MAML":
                self.model = GraphClassifier(backbone_dim_in=info["input_size"], backbone_dim_h=info["hidden_size"],
                                             backbone_dim_out=info["output_size"], backbone_num_layers=info["num_layers"], 
                                            projection = info["projection"], num_classes=self.opt["settings"]["few_shot"]["train"]["class_per_iter"],
                                            backbone_type='gcn')
            else:
                self.model = GCN(dim_in=info["input_size"], dim_h=info["hidden_size"], 
                                dim_out=info["output_size"], num_layers=info["num_layers"], 
                                projection = info["projection"])
        elif info["model_name"] == "GAT":
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GATLayer(dim_in=info["input_size"], dim_h=info["hidden_size"], 
                                      dim_o=info["output_size"], num_layers=info["num_layers"])
        elif info["model_name"] == "GIN":
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GINLayer(dim_in=info["input_size"], dim_h=info["hidden_size"], 
                                      dim_o=info["output_size"], num_layers=info["num_layers"])
            else:
                self.model = GIN(dim_in=info["input_size"], dim_h=info["hidden_size"], 
                                dim_out=info["output_size"], num_layers=info["num_layers"], 
                                projection = info["projection"])
        else:
            raise ValueError("Model not supported")
        
        if torch.cuda.is_available() and self.device == "cpu":
            print("CUDA is available, but you are using CPU")
        print(f"Device: {self.device}")
        record_log(self.log_file, f"Device: {self.device}\n")

        if self.parallel and len(self.parallel_device) > 1:
            self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=self.parallel_device) 
        elif self.parallel:
            print("You didn't specify the device ids for parallel training")
            record_log(self.log_file, "You didn't specify the device ids for parallel training\n")
            print("Using all available devices", torch.cuda.device_count(), "GPUs")       
            record_log(self.log_file, f"Using all available devices {torch.cuda.device_count()} GPUs\n")
            self.model = torch.nn.DataParallel(self.model.to(self.device))
        else:
            self.model = self.model.to(self.device)
    def get_loss_fn(self):
        if self.opt["settings"]["few_shot"]["method"] == "ProtoNet":
            loss_fn = ProtoLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "NnNet":
            loss_fn = NnLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "SoftNnNet":
            loss_fn = SoftNnLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
            loss_fn = LabelPropagation(self.opt, self.model)
        elif self.opt["settings"]["few_shot"]["method"] == "MatchNet":
            loss_fn = MatchLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "RelationNetwork":
            loss_fn = RelationNetwork(self.opt, self.model)
        elif self.opt["settings"]["few_shot"]["method"] == "MAML":
            loss_fn = MAMLLoss(self.opt, self.model)
        else:
            raise ValueError("Loss method not supported")
        self.loss_fn = loss_fn
    def get_optimizer(self):
        if self.opt["settings"]["train"]["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt["settings"]["train"]["lr"])
        elif self.opt["settings"]["train"]["optimizer"] == "AdamW":
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opt["settings"]["train"]["lr"])
        elif self.opt["settings"]["train"]["optimizer"] == "SGD":
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt["settings"]["train"]["lr"])
        else:
            raise ValueError("Optimizer not supported")
        self.optim = optimizer
        
    def get_lr_scheduler(self):
        if self.opt["settings"]["train"]["lr_scheduler"]["method"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=self.opt["settings"]["train"]["lr_scheduler"]["step_size"], gamma=self.opt["settings"]["train"]["lr_scheduler"]["gamma"])
        elif self.opt["settings"]["train"]["lr_scheduler"]["method"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=self.opt["settings"]["train"]["lr_scheduler"]["factor"], 
                                                                   patience=self.opt["settings"]["train"]["lr_scheduler"]["patience"], 
                                                                   min_lr=self.opt["settings"]["train"]["lr_scheduler"]["min_lr"],
                                                                   cooldown=self.opt["settings"]["train"]["lr_scheduler"]["cooldown"])
        else:
            raise ValueError("LR scheduler not supported")
        self.scheduler = scheduler

    def setting(self):
        print("Setting up the training module...")
        print(f"Loading data from {self.embeddingFolder}...")
        print("Loading training data...")
        self.trainGraph, label = load_GE_data(self.trainDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "trainData.pkl"))
        
        if self.enable_openset:
            assert(self.openset_m_samples <= self.support_shots_train + self.query_shots_train)
            sampler = FcgSampler(label, self.support_shots_train + self.query_shots_train, self.class_per_iter_train + self.openset_class_per_iter, self.iterations)
        else:
            sampler = FcgSampler(label, self.support_shots_train + self.query_shots_train, self.class_per_iter_train, self.iterations)
        self.trainLoader = DataLoader(self.trainGraph, batch_sampler=sampler, num_workers=4, collate_fn=collate_graphs)    
        if self.valDataset is not None:
            print("Loading validation data...")
            self.valGraph, label =load_GE_data(self.valDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "valData.pkl"))
            if self.enable_openset:
                assert(self.opensetData is not None)
                print("Generating open set testing data...")
                print("Loading openset data...")
                if self.opt["dataset"]["openset_data_mode"] == "random":
                    ratio = self.opt["dataset"]["openset_data_ratio"]
                    opensetPklName = f"opensetData_random_{ratio}.pkl"
                    info = f"random_{ratio}"
                else:
                    opensetPklName = "opensetData.pkl"
                    info = ""
                opensetGraph, _ = load_GE_data(self.opensetData, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, opensetPklName), openset=True, opensetInfo=info)
                # self.opensetLoader = DataLoader(opensetGraph, batch_size=20, shuffle=True, num_workers=4, collate_fn=collate_graphs)
                
                opensetSampler = OpenSetFcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations, opensetGraph, self.openset_m_samples_test)
                self.valLoader = DataLoader(ConcatDataset([self.valGraph, opensetGraph]), batch_sampler=opensetSampler, num_workers=4, collate_fn=collate_graphs)    
            else:
                val_sampler = FcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations)
                self.valLoader = DataLoader(self.valGraph, batch_sampler=val_sampler, num_workers=4, collate_fn=collate_graphs)
        else:
            self.valLoader = None

        self.get_backbone()
        self.get_loss_fn()
        if self.opt["settings"]["few_shot"]["method"] in ("LabelPropagation", "RelationNetwork", "MAML"):
            self.model = self.loss_fn
        self.get_optimizer()
        if self.lr_scheduler:
            self.get_lr_scheduler()

        info = self.opt["settings"]["model"]
        if info["load_weights"]:
            model_path = os.path.abspath(info["load_weights"])
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            # 將優化器狀態移動到正確的設備
            for param_group in self.optim.param_groups:
                for param in param_group['params']:
                    # 確保參數在正確的設備上
                    if param.device != self.device:
                        param.data = param.data.to(self.device)
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(self.device)
            
            # 將優化器內部狀態移動到正確的設備
            for state in self.optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.model = self.model.to(self.device)
            print(f"Model loaded from {model_path}")
            record_log(self.log_file, f"Model loaded from {model_path}\n")
        
        print(f"Model: {self.model}")
        record_log(self.log_file, f"Model: {self.model}\n")
        print(f"Loss function: {self.loss_fn}")
        record_log(self.log_file, f"Loss function: {self.loss_fn}\n")
        print(f"Optimizer: {self.optim}") 
        record_log(self.log_file, f"Optimizer: {self.optim}\n")
        print("Finish setting up the training module")

    def train(self):
        print("Copying split files...")
        splitFolder = self.opt["paths"]["data"]["split_folder"]
        splitFiles = [f for f in os.listdir(splitFolder) if f.endswith(f"{self.datasetName}.txt")]
        for f in splitFiles:
            src = os.path.join(splitFolder, f)
            dst = os.path.join(os.path.dirname(self.model_folder), f)
            if not os.path.exists(dst):
                os.system(f"cp {src} {dst}")
        print("Finish copying split files")
    
        print("Start training...")
        record_log(self.log_file, "Start training...\n")
        self.run()
        print("Finish training")
        record_log(self.log_file, "Finish training\n")

            
class TestModule(Testing):
    def __init__(self, configPath: str, dataset: LoadDataset):
        
        # opt = load_config(configPath)
        self.model_folder = os.path.dirname(configPath)
        opt = load_config(configPath)
        self.opt = opt
        self.embeddingFolder = os.path.join(opt["paths"]["data"]["embedding_folder"], dataset.datasetName, opt["settings"]["vectorize"]["node_embedding_method"])
        self.testDataset = dataset.testData
        self.valDataset = dataset.valData
        self.opensetData = dataset.opensetData
        self.loss_fn = None

        self.support_shots_test = opt["settings"]["few_shot"]["test"]["support_shots"]
        self.query_shots_test = opt["settings"]["few_shot"]["test"]["query_shots"]
        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("test", {}).get("use", False)
        self.openset_m_samples = opt["settings"]["openset"]["test"]["m_samples"] if self.enable_openset else 0
        self.class_per_iter_test = opt["settings"]["few_shot"]["test"]["class_per_iter"]

        self.iterations = opt["settings"]["train"]["iterations"]
        self.device = opt["settings"]["train"]["device"]
        self.embeddingSize = opt["settings"]["vectorize"]["node_embedding_size"]

        self.pretrain_folder = opt["paths"]["model"]["pretrained_folder"]
        self.setting()

        super().__init__(
            device=self.device,
            loss_fn=self.loss_fn,
            openset=self.enable_openset,
        )
        
    def get_loss_fn(self):
        if self.opt["settings"]["few_shot"]["method"] == "ProtoNet":
            loss_fn = ProtoLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "NnNet":
            loss_fn = NnLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "SoftNnNet":
            loss_fn = SoftNnLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
            loss_fn = LabelPropagation(self.opt, self.model)
        elif self.opt["settings"]["few_shot"]["method"] == "MatchNet":
            loss_fn = MatchLoss(self.opt)
        elif self.opt["settings"]["few_shot"]["method"] == "RelationNetwork":
            loss_fn = RelationNetwork(self.opt, self.model)
        elif self.opt["settings"]["few_shot"]["method"] == "MAML":
            loss_fn = MAMLLoss(self.opt, self.model)
        else:
            raise ValueError("Loss method not supported")
        self.loss_fn = loss_fn
    def generate_model(self, opt: dict = None):
        info = self.opt["settings"]["model"]
        if info["model_name"] == "GraphSAGE":
            self.pretrainModel = GraphSAGE(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GraphSAGELayer(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_o=info["output_size"], num_layers=info["num_layers"])
            elif self.opt["settings"]["few_shot"]["method"] == "MAML":
                self.model = GraphClassifier(backbone_dim_in=info["input_size"], backbone_dim_h=info["hidden_size"],
                                             backbone_dim_out=info["output_size"], backbone_num_layers=info["num_layers"], 
                                            projection = info["projection"], num_classes=self.opt["settings"]["few_shot"]["test"]["class_per_iter"],
                                            backbone_type='sage')
            else:
                self.model = GraphSAGE(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
        elif info["model_name"] == "GCN":
            self.pretrainModel = GCN(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GCNLayer(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_o=info["output_size"], num_layers=info["num_layers"])
            elif self.opt["settings"]["few_shot"]["method"] == "MAML":
                self.model = GraphClassifier(backbone_dim_in=info["input_size"], backbone_dim_h=info["hidden_size"],
                                             backbone_dim_out=info["output_size"], backbone_num_layers=info["num_layers"], 
                                            projection = info["projection"], num_classes=self.opt["settings"]["few_shot"]["test"]["class_per_iter"],
                                            backbone_type='gcn')
            else:
                self.model = GCN(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
        elif info["model_name"] == "GAT":
            ### pretrain_model
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GATLayer(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_o=info["output_size"], num_layers=info["num_layers"])
        elif info["model_name"] == "GIN":
            self.pretrainModel = GIN(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
            if self.opt["settings"]["few_shot"]["method"] == "LabelPropagation":
                self.model = GINLayer(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_o=info["output_size"], num_layers=info["num_layers"])
            else:
                self.model = GIN(dim_in=info["input_size"], dim_h=info["hidden_size"], dim_out=info["output_size"], num_layers=info["num_layers"], projection = info["projection"])
        else:
            raise ValueError("Model not supported")
    
    def setting(self):
        print("Setting up the testing module...")
        testGraph, label = load_GE_data(self.testDataset, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, "testData.pkl"))
        print(f"Loading data from {self.embeddingFolder}...")
        if self.opensetData is not None and self.enable_openset:
            print("Generating open set testing data...")
            print("Loading openset data...")
            if self.opt["dataset"]["openset_data_mode"] == "random":
                ratio = self.opt["dataset"]["openset_data_ratio"]
                opensetPklName = f"opensetData_random_{ratio}.pkl"
                info = f"random_{ratio}"
            else:
                opensetPklName = "opensetData.pkl"
                info = ""
            opensetGraph, _ = load_GE_data(self.opensetData, self.embeddingFolder, self.embeddingSize, os.path.join(self.embeddingFolder, opensetPklName), openset=True, opensetInfo=info)
            # self.opensetLoader = DataLoader(opensetGraph, batch_size=20, shuffle=True, num_workers=4, collate_fn=collate_graphs)
            
            opensetSampler = OpenSetFcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations, opensetGraph, self.openset_m_samples)
            self.opensetTestLoader = DataLoader(ConcatDataset([testGraph, opensetGraph]), batch_sampler=opensetSampler, num_workers=4, collate_fn=collate_graphs)    
        else:
            print("Generating closed set testing data...")
            print("Loading testing data...")
            sampler = FcgSampler(label, self.support_shots_test + self.query_shots_test, self.class_per_iter_test, self.iterations)
            self.testLoader = DataLoader(testGraph, batch_sampler=sampler, num_workers=4, collate_fn=collate_graphs)    

        self.generate_model()
        self.get_loss_fn()
        
        if self.opt["settings"]["few_shot"]["method"] in ("LabelPropagation", "RelationNetwork", "MAML"):
            self.model = self.loss_fn

        print("Finish setting up the testing module")
    
    def load_best_model(self):
        model_path = os.path.join(self.model_folder, [f for f in os.listdir(self.model_folder) if "best" in f][0])
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        print(f"Model loaded from {model_path}")
        self.model = self.model.to(self.device)
    
    def eval(self, model_path: str = None):

        """
        Evaluate the model on the test dataset.
        Args:
            model_path (str): Path to the model file. If None, the best model will be used.
            mode (str): Evaluation mode, either "closedset" or "openset".
        """
    
        if model_path is None:
            print("Model path is not provided. Using the best model...")
            model_path = os.path.join(self.model_folder, [f for f in os.listdir(self.model_folder) if "best" in f][0])
            model_name = os.path.basename(model_path)
        else:
            model_name = os.path.dirname(model_path).split("/")[-1] + "_" + os.path.basename(model_path)
        evalFolder = os.path.dirname(model_path)
        logFolder = os.path.dirname(self.model_folder)
        
        if self.enable_openset == False:
            assert(self.loss_fn.enable_openset == False, "Open set evaluation is not supported in this mode")

            print("Record evaluation log...")
            evalLogPath = os.path.join(logFolder, "evalLog.csv")
            if not os.path.exists(evalLogPath):
                with open(evalLogPath, "w") as f:
                    f.write("timestamp, folderName, model, test_acc\n")
        else: # "openset"
            assert(self.loss_fn.enable_openset == True, "You need to enable openset in the model") 
            assert(self.opensetTestLoader is not None, "You need to load openset data")
            
            print("Record evaluation log...")
            evalLogPath = os.path.join(logFolder, "evalLog_openset.csv")
            if not os.path.exists(evalLogPath):
                with open(evalLogPath, "w") as f:
                    f.write("timestamp, folderName, model, closedset_test_acc, openset_auroc \n")
    
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
        self.model = self.model.to(self.device)
        
        print(f"Model: {self.model}")
        print(f"Start evaluation... (testing dataset)")
        if self.enable_openset == False: 
            testAcc = self.testing(self.model, self.testLoader)
        else:
            print("Open set evaluation enabled")
            print(f"Open set m_samples: {self.openset_m_samples}")
            testAcc, testAuc = self.testing(self.model, self.opensetTestLoader)
        if os.path.basename(evalFolder).split("_")[0] == "10way" and self.opt["settings"]["few_shot"]["test"]["class_per_iter"] == 5:
            folder_name = os.path.basename(evalFolder) + "_test_in_5way"
        else:
            folder_name = os.path.basename(evalFolder)
        if self.enable_openset == False:
            print(f"Closed set test accuracy: {testAcc}")
            with open(evalLogPath, "a") as f:
                f.write(f"{datetime.now()}, {folder_name}, {model_name}, {testAcc}\n")
        else:
            print(f"Closed set test accuracy: {testAcc}")
            print(f"Open set AUROC: {testAuc}")
            with open(evalLogPath, "a") as f:
                f.write(f"{datetime.now()}, {folder_name}, {model_name}, {testAcc}, {testAuc}\n")
            
        print("Finish evaluation")

    def eval_ablation(self, model_path: str = None): ## Just eval ablation part
        if model_path is None:
            print("Model path is not provided. Using the best model...")
            model_path = os.path.join(self.model_folder, [f for f in os.listdir(self.model_folder) if "best" in f][0])

        evalFolder = os.path.dirname(model_path)
        logFolder = os.path.dirname(self.model_folder)

        print("Record evaluation log...")
        evalLogPath = os.path.join(logFolder, "evalLog.csv")
        if not os.path.exists(evalLogPath):
            with open(evalLogPath, "w") as f:
                f.write("timestamp, folderName, model, test_acc, val_acc\n")
    
        if self.opt["settings"]["model"]["load_weights"] != "":
            pretrainModelFolder = os.path.join(self.pretrain_folder, self.opt["settings"]["model"]["load_weights"])
            pretrainModelPath = os.path.join(pretrainModelFolder, [f for f in os.listdir(pretrainModelFolder) if "best_backbone" in f][0])
            self.pretrainModel.load_state_dict(torch.load(pretrainModelPath, map_location=self.device)["model_state_dict"], strict=False)
        else:
            pretrainModelPath = "None"

        print(f"Ablation evaluation... (testing dataset)")
        self.pretrainModel = self.pretrainModel.to(self.device)
        
        testAccPretrain = self.testing(self.pretrainModel, self.testLoader)
        if os.path.basename(evalFolder).split("_")[0] == "10way" and self.opt["settings"]["few_shot"]["test"]["class_per_iter"] == 5:
            folder_name = os.path.basename(evalFolder) + "_test_in_5way"
        with open(evalLogPath, "a") as f:
            f.write(f"{datetime.now()}, {folder_name}, {os.path.basename(pretrainModelPath)}, {testAccPretrain}\n")
            
        print("Finish evaluation")