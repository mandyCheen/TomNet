import os
from utils import save_checkpoint, save_config, record_log, save_model_architecture
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import pickle
from torch_geometric.utils.convert import from_networkx
import sklearn.preprocessing as labelEncoder
from datetime import datetime

def load_GE_data(dataset: pd.DataFrame, embeddingFolder: str, embeddingSize: int, dataPath: str, openset: bool = False, opensetInfo: str = None):
    if not os.path.exists(dataPath):
        labels = []
        graphList = []
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            cpu = row["CPU"]
            family = row["family"]
            fileName = row["file_name"]
            filePath = f"{embeddingFolder}/{cpu}/{family}/{fileName}.gpickle"
            with open(filePath, "rb") as f:
                fcg = pickle.load(f)
            for node in fcg.nodes:
                if len(fcg.nodes[node]["x"]) == 0:
                    fcg.nodes[node]["x"] = torch.zeros(embeddingSize, dtype=torch.float32)
                if not isinstance(fcg.nodes[node]["x"], torch.Tensor):
                    fcg.nodes[node]["x"] = torch.tensor(fcg.nodes[node]["x"], dtype=torch.float32)
            torch_data = from_networkx(fcg)
            labels.append(family)
            graphList.append(torch_data)
        le = labelEncoder.LabelEncoder()
        le.fit(labels)
        labels_ = le.transform(labels)
        folder = os.path.dirname(dataPath)
        labelDict = dict(zip(le.classes_, le.transform(le.classes_)))
        if openset:
            labelDictName = f"labelDict_openset_{opensetInfo}.pkl"
        else:
            labelDictName = "labelDict.pkl"
        with open(f"{folder}/{labelDictName}", "wb") as f:
            pickle.dump(labelDict, f)
        for i, data in enumerate(graphList):
            data.y = torch.tensor(labels_[i])
        print(f"Saving data to {dataPath}")
        with open(dataPath, "wb") as f:
            pickle.dump([graphList, labels_], f)
    else:
        print(f"Loading data from {dataPath}...")
        with open(dataPath, "rb") as f:
            graphList, labels_ = pickle.load(f)
    
    return graphList, labels_

class Training:
    def __init__(self, opt, trainLoader, valLoader, model, loss_fn, optim, scheduler, device, model_path):

        self.opt = opt
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.scheduler = scheduler
        self.device = device

        self.epochs = opt["settings"]["train"]["num_epochs"]
        self.iterations = list(range(self.epochs * opt["settings"]["train"]["iterations"]))
        self.save_model = opt["settings"]["train"]["save_model"]
        self.early_stopping = opt["settings"]["train"]["early_stopping"]["use"]
        self.early_stopping_patience = opt["settings"]["train"]["early_stopping"]["patience"]

        self.enable_openset_training = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)

        self.model_folder = model_path     
        os.makedirs(self.model_folder, exist_ok=True)   
        self.log_file = self.model_folder + "/log.txt"

        self.train_acc_history = []
        self.val_acc_history = []

    def end_of_epoch(self, avg_acc, best_acc, epoch, patience, avg_loss):
        if self.scheduler:
            if self.opt["settings"]["train"]["lr_scheduler"]["method"] == "ReduceLROnPlateau":
                self.scheduler.step(avg_loss)
            elif self.opt["settings"]["train"]["lr_scheduler"]["method"] == "StepLR":
                self.scheduler.step()
            print(f"Current learning rate: {self.scheduler.get_last_lr()}")
            record_log(self.log_file, f"Current learning rate: {self.scheduler.get_last_lr()}\n")
        if avg_acc >= best_acc:
            best_acc = avg_acc
            if self.save_model:
                self.best_model = self.model
                save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=True, epoch=epoch+1, checkpoint=self.model_folder, value=best_acc)
            if self.early_stopping:
                patience = 0
        else:
            if self.early_stopping:
                patience += 1
                if patience == self.early_stopping_patience:
                    print("Early stopping")
                    record_log(self.log_file, "Early stopping in epoch {}\n".format(epoch+1))
                    return best_acc, patience, True
        if self.save_model and (epoch+1) % 10 == 0:
            save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=False, epoch=epoch+1, checkpoint=self.model_folder, value=avg_acc)
        if self.early_stopping:
            print(f"Patience: {patience}/{self.early_stopping_patience}")
            record_log(self.log_file, f"Patience: {patience}/{self.early_stopping_patience}\n")
        torch.cuda.empty_cache()
        return best_acc, patience, False

    def end_of_epoch_loss(self, avg_loss, lowest_loss, epoch, patience):
        if self.scheduler:
            if self.opt["settings"]["train"]["lr_scheduler"]["method"] == "ReduceLROnPlateau":
                self.scheduler.step(avg_loss)
            elif self.opt["settings"]["train"]["lr_scheduler"]["method"] == "StepLR":
                self.scheduler.step()
            print(f"Current learning rate: {self.scheduler.get_last_lr()}")
            record_log(self.log_file, f"Current learning rate: {self.scheduler.get_last_lr()}\n")
        if avg_loss <= lowest_loss:
            lowest_loss = avg_loss
            if self.save_model:
                self.best_model = self.model
                save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=True, epoch=epoch+1, checkpoint=self.model_folder, value=lowest_loss)
            if self.early_stopping:
                patience = 0
        else:
            if self.early_stopping:
                patience += 1
                if patience == self.early_stopping_patience:
                    print("Early stopping")
                    record_log(self.log_file, "Early stopping in epoch {}\n".format(epoch+1))
                    return lowest_loss, patience, True
        if self.save_model and (epoch+1) % 10 == 0:
            save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=False, epoch=epoch+1, checkpoint=self.model_folder, value=avg_loss)
        if self.early_stopping:
            print(f"Patience: {patience}/{self.early_stopping_patience}")
            record_log(self.log_file, f"Patience: {patience}/{self.early_stopping_patience}\n")
        torch.cuda.empty_cache()
        return lowest_loss, patience, False
    
    def end_of_epoch_pretrain(self, avg_acc, best_acc, epoch, patience, avg_loss):
        if self.scheduler:
            if self.opt["settings"]["train"]["lr_scheduler"]["method"] == "ReduceLROnPlateau":
                self.scheduler.step(avg_loss)
            elif self.opt["settings"]["train"]["lr_scheduler"]["method"] == "StepLR":
                self.scheduler.step()
            print(f"Current learning rate: {self.scheduler.get_last_lr()}")
        if avg_acc >= best_acc:
            best_acc = avg_acc
            if self.save_model:
                self.best_model = self.model
                save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=True, epoch=epoch+1, checkpoint=self.model_folder, value=best_acc)
                save_checkpoint(model_state=self.model.backbone.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=True, epoch=epoch+1, checkpoint=self.model_folder, backbone=True, value=best_acc)
            if self.early_stopping:
                patience = 0
        else:
            if self.early_stopping:
                patience += 1
                if patience == self.early_stopping_patience:
                    print("Early stopping")
                    record_log(self.log_file, "Early stopping in epoch {}\n".format(epoch+1))
                    return best_acc, patience, True
        if self.save_model and (epoch+1) % 10 == 0:
            save_checkpoint(model_state=self.model.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=False, epoch=epoch+1, checkpoint=self.model_folder, value=avg_acc)
            save_checkpoint(model_state=self.model.backbone.state_dict(), optim_state=self.optim.state_dict(), sche_state=self.scheduler.state_dict(), is_best=False, epoch=epoch+1, checkpoint=self.model_folder, backbone=True, value=avg_acc)
        if self.early_stopping:
            print(f"Patience: {patience}/{self.early_stopping_patience}")
            record_log(self.log_file, f"Patience: {patience}/{self.early_stopping_patience}\n")
        torch.cuda.empty_cache()
        return best_acc, patience, False   

    def run(self):
        best_train_acc = 0
        best_val_acc = 0
        lowest_train_loss = 1000
        lowest_val_loss = 1000
        patience = 0
        stop = False
        print(f"Current learning rate: {self.scheduler.get_last_lr()}")
        # save config
        save_config(self.opt, self.model_folder + "/config.json")
        # save model architecture
        save_model_architecture(self.model, self.model_folder + "/model_architecture.txt")
        for epoch in range(self.epochs):
            self.model.train()
            train_acc = []
            train_loss = []
            train_auc = []
            val_acc = []
            val_loss = []
            val_auc = []
            with tqdm(self.trainLoader, desc=f"Epoch {epoch+1}/{self.epochs} (Training)") as pbar:
                for data in pbar:
                    data = data.to(self.device)
                    self.optim.zero_grad()
                    if self.opt["settings"]["few_shot"]["method"] in ("LabelPropagation", "RelationNetwork", "MAML"):
                        loss, acc = self.model(data)
                    else:
                        predicts = self.model(data)
                        loss, acc = self.loss_fn(predicts, data.y)    
                    loss.backward()
                    self.optim.step()   
                    self.train_acc_history.append(acc.item())
                    # Update progress bar with current batch metrics
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{acc.item():.4f}',
                        'openset_auc': f'{self.model.openset_auroc:.4f}' if self.enable_openset_training else 'N/A'
                    })
                    train_loss.append(loss.item())
                    train_acc.append(acc.item()) 
                    if self.enable_openset_training:
                        train_auc.append(self.model.openset_auroc)
                avg_loss = np.mean(train_loss)
                avg_acc = np.mean(train_acc)
                if self.enable_openset_training:
                    avg_auc = np.mean(train_auc)
                postfix = ' (Best)' if avg_acc >= best_train_acc else f' (Best: {best_train_acc:.4f})'
                # postfix = ' (Lowest)' if avg_loss <= lowest_train_loss else f' (Lowest: {lowest_train_loss:.4f})'
                content = f'Avg Train Loss: {avg_loss:.4f}, Avg Train Acc: {avg_acc:.4f}{postfix}'
                # content = f'Avg Train Loss: {avg_loss:.4f}{postfix}, Avg Train Acc: {avg_acc:.4f}'
                if self.valLoader is not None and avg_acc >= best_train_acc:
                    best_train_acc = avg_acc
                # if self.valLoader is not None and avg_loss <= lowest_train_loss:
                #     lowest_train_loss = avg_loss
                print(content)
                record_log(self.log_file, f"Epoch {epoch+1}/{self.epochs}: {content}\n")
                if self.enable_openset_training:
                    record_log(self.log_file, 'Open-Set AUROC: {:.4f}\n'.format(avg_auc))
                
            if self.valLoader is not None: 
                self.model.eval()                
                with tqdm(self.valLoader, desc=f"Epoch {epoch+1}/{self.epochs} (Validation)") as pbar:
                    for data in pbar:
                        data = data.to(self.device)
                        with torch.no_grad():
                            if self.opt["settings"]["few_shot"]["method"] in ("LabelPropagation", "RelationNetwork", "MAML"):
                                loss, acc = self.model(data, opensetTesting=self.enable_openset_training)
                            else:
                                model_output = self.model(data)
                                loss, acc = self.loss_fn(model_output, data.y)
                        val_loss.append(loss.item())
                        val_acc.append(acc.item())
                        self.val_acc_history.append(acc.item())
                        # Update progress bar with current batch metrics
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{acc.item():.4f}',
                            'openset_auc': f'{self.model.openset_auroc:.4f}' if self.enable_openset_training else 'N/A'
                        })
                    avg_loss = np.mean(val_loss)
                    avg_acc = np.mean(val_acc)
                    if self.enable_openset_training:
                        avg_auc = np.mean(val_auc)
                    postfix = ' (Best)' if avg_acc >= best_val_acc else f' (Best: {best_val_acc:.4f})'
                    # postfix = ' (Lowest)' if avg_loss <= lowest_val_loss else f' (Lowest: {lowest_val_loss:.4f})'
                    content = f'Avg Val Loss: {avg_loss:.4f}, Avg Val Acc: {avg_acc:.4f}{postfix}'
                    # content = f'Avg Val Loss: {avg_loss:.4f}{postfix}, Avg Val Acc: {avg_acc:.4f}'
                    print(content)
                    record_log(self.log_file, f"Epoch {epoch+1}/{self.epochs}: {content}\n")
                    if self.enable_openset_training:
                        record_log(self.log_file, 'Open-Set AUROC: {:.4f}\n'.format(avg_auc))
                    best_val_acc, patience, stop = self.end_of_epoch(avg_acc, best_val_acc, epoch, patience, avg_loss)
                    # lowest_val_loss, patience, stop = self.end_of_epoch_loss(avg_loss, lowest_val_loss, epoch, patience)
            else:
                best_train_acc, patience, stop = self.end_of_epoch(avg_acc, best_train_acc, epoch, patience, avg_loss)
                # lowest_train_loss, patience, stop = self.end_of_epoch_loss(avg_loss, lowest_train_loss, epoch, patience)
            if stop:
                break
        return True

class Testing:
    def __init__(self, device, loss_fn, openset=False):
        self.device = device
        self.loss_fn = loss_fn
        self.openset = openset
    
    def testing(self, testModel, testLoader):
        avg_acc = list()
        avg_auc = list()
        for epoch in range(10):
            print(f"Epoch {epoch+1}")
            with tqdm(testLoader, desc=f"Epoch {epoch+1}/{10} (Testing)") as pbar:
                for data in pbar:
                    testModel.eval()
                    # print(np.unique(data.y))
                    data = data.to(self.device)
                    with torch.no_grad():
                        if self.opt["settings"]["few_shot"]["method"] in ("LabelPropagation", "RelationNetwork", "MAML"):
                            loss, acc = testModel(data, opensetTesting=self.openset)
                            if self.openset:
                                avg_auc.append(testModel.openset_auroc)
                                opauroc = testModel.openset_auroc
                        else:
                            model_output = testModel(data)
                            loss, acc = self.loss_fn(model_output, data.y, opensetTesting=self.openset)
                            if self.openset:
                                avg_auc.append(self.loss_fn.openset_auroc)
                                opauroc = self.loss_fn.openset_auroc
                        avg_acc.append(acc.item())
                    # Update progress bar with current batch metrics
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{acc.item():.4f}',
                        'openset_auc': f'{opauroc:.4f}' if self.openset else 'N/A'
                    })
            torch.cuda.empty_cache()
                    
        avg_acc = np.mean(avg_acc)
        if self.openset:
            avg_auc = np.mean(avg_auc)
            print(f"Testing accuracy: {avg_acc:.4f}, OpenSet AUROC: {avg_auc:.4f}")
            return avg_acc, avg_auc
        else:
            print(f"Testing accuracy: {avg_acc:.4f}")
            return avg_acc

            