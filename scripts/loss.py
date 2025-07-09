import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from collections import OrderedDict
import numpy as np
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from scripts.models import GraphRelationNetwork, MLPRelationModule
from torch_geometric.nn import global_add_pool
import copy
from torch_geometric.data import Data, Batch

class DistanceMetric:
    """Base class for distance metrics"""
    @staticmethod
    def compute(x, y):
        raise NotImplementedError

class LossFunction:
    """Base class for loss functions"""
    @staticmethod
    def compute(dists, n_classes, n_query):
        raise NotImplementedError
    @staticmethod
    def compute(dists, n_classes, n_query, negative_distance=False):
        raise NotImplementedError
    @staticmethod
    def compute(matrix, dim):
        raise NotImplementedError

class EuclideanDistance(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute euclidean distance between two tensors
        x: N x D
        y: M x D
        '''
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception("Dimension mismatch")

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

class CosineDistance(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute cosine distance between two tensors
        x: N x D
        y: M x D
        '''
        # 正規化向量
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        
        # 計算余弦相似度
        similarity = torch.mm(x_norm, y_norm.t())
        
        # 轉換為距離（1 - similarity）
        return 1 - similarity

class ManhattanDistance(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute Manhattan (L1) distance between two tensors
        x: N x D
        y: M x D
        '''
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception("Dimension mismatch")

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.abs(x - y).sum(2)

class CosineSimilarity(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute cosine similarity between two tensors
        x: N x D
        y: M x D
        '''
        # 正規化向量
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        
        # 計算余弦相似度
        similarity = torch.mm(x_norm, y_norm.t())
        
        return similarity
    
class CrossEntropyLoss(LossFunction):
    @staticmethod
    def compute(dists, n_classes, n_query, negative_distance=False):
        """
        使用CrossEntropyLoss計算損失和準確率
        
        Args:
            dists: 距離矩陣
            n_classes: 類別數量
            n_query: 每個類別的查詢樣本數
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        # 將距離轉換為logits (負距離作為logits)
        if negative_distance:
            logits = -dists.view(n_classes, n_query, -1)
        else:
            logits = dists.view(n_classes, n_query, -1)
        
        # 準備目標索引
        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()
        
        # 計算預測類別
        # 使用softmax而不是log_softmax來獲得概率
        probs = F.softmax(logits, dim=2)
        _, y_hat = probs.max(2)
        
        # 重塑logits和目標
        logits = logits.view(-1, n_classes)  # (n_classes * n_query) x n_classes
        target_ = target_inds.reshape(-1)    # (n_classes * n_query)
        
        # 使用CrossEntropyLoss (內部包含softmax)
        criterion = nn.CrossEntropyLoss()
        loss_val = criterion(logits, target_)
        
        # 計算準確率
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
        
        return loss_val, acc_val
    
class EntropyLoss(LossFunction):
    """熵損失計算類"""

    @staticmethod
    def compute(matrix, dim):
        """
        計算熵損失
        
        Args:
            matrix: 輸入矩陣
            dim: 計算熵的維度
            
        Returns:
            torch.Tensor: 熵損失值
        """
        # 計算每行的softmax
        probs = F.softmax(matrix, dim=dim)
        log_probs = F.log_softmax(matrix, dim=dim)
        
        entropy_loss = -torch.sum(probs * log_probs, dim=dim).mean()

        return entropy_loss
        

DISTANCE_METRICS = {
    'euclidean': EuclideanDistance,
    'cosine_distance': CosineDistance,
    'cosine_similarity': CosineSimilarity,
    'manhattan': ManhattanDistance
}

LOSS_FUNCTIONS = {
    'CrossEntropyLoss': CrossEntropyLoss
}

class Loss:
    def __init__(self, n_support):
        self.n_support = n_support
        self.openset_auroc = 0.0
        
    def compute_prototypes(self, input, support_idxs):
        """Compute prototype vectors"""
        return torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
    
    def get_nn_points(self, input, support_idxs):
        """Get nearest neighbor points"""
        return torch.cat([input[idx_list] for idx_list in support_idxs])
    
    def get_support_query_idxs(self, target, openset=False, cls_openset=None):
        """Split data into support and query sets"""
        def supp_idxs(c):
            return target.eq(c).nonzero()[:self.n_support].squeeze(1)
        classes = torch.unique(target)
        cls_num = len(classes)
        if openset:
            assert cls_openset is not None, "cls_support must be provided for openset"
            cls_support = cls_num - cls_openset
            # Randomly select classes for support
            random_classes = classes[torch.randperm(len(classes))]
            support_idxs = list(map(supp_idxs, random_classes[:cls_support]))
            query_idxs = torch.stack(list(map(
                lambda c: target.eq(c).nonzero()[self.n_support:], random_classes[:cls_support]
            ))).view(-1)
            openset_idxs = torch.stack(list(map(
                lambda c: target.eq(c).nonzero()[:], random_classes[cls_support:]
            ))).view(-1)
            return random_classes, support_idxs, query_idxs, openset_idxs
        else:
            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(
                lambda c: target.eq(c).nonzero()[self.n_support:], classes
            ))).view(-1)
        
            return classes, support_idxs, query_idxs

    def reshape_random_data(self, data, target, n_ways, n_shot, n_queries):
        """
        Reshape randomly organized data into (n_ways, n_shot + n_queries, n_features) format
        Returns:
            Reshaped data of shape (n_ways, n_shot + n_queries, n_features)
        """
        # Set n_support for the get_support_query_idxs function
        n_support = n_shot
        
        # Get classes, support indices, and query indices
        classes, support_idxs, query_idxs = self.get_support_query_idxs(target)
        
        # Get feature dimension
        n_features = data.shape[1]
        
        # Initialize the result tensor
        result = torch.zeros(n_ways, n_shot + n_queries, n_features)
        
        # Fill in support samples
        for i, c in enumerate(classes):
            # Get support samples for this class
            class_support = data[support_idxs[i]]
            # Place them in the result tensor
            result[i, :n_shot] = class_support
        
        # Fill in query samples
        query_data = data[query_idxs]
        # Reshape query data to (n_ways, n_queries, n_features)
        query_data = query_data.reshape(n_ways, n_queries, n_features)
        # Place them after the support samples
        result[:, n_shot:] = query_data
        
        return result

    def get_avg_distance_between_support(self, input, target):
        """
        Compute average distance between support samples of the same class
        """

        class_avg_distances = []

        _, support_idxs, query_idxs = self.get_support_query_idxs(target)

        for class_idx, idx_list in enumerate(support_idxs):
            # Get support samples for this class
            class_support_samples = input[idx_list]
                
            # Compute pairwise distances within this class
            class_dists = self.distance_metric.compute(class_support_samples, class_support_samples)
            
            # Create a mask to exclude self-distances (diagonal elements)
            mask = ~torch.eye(len(idx_list), dtype=torch.bool, device=class_dists.device)
            
            # Calculate mean of non-diagonal elements (actual distances between different samples)
            class_avg_dist = class_dists[mask].mean()
            print(f"Class {class_idx}: Average distance = {class_avg_dist.item()}")
            class_avg_distances.append(class_avg_dist)
        
        # Convert to tensor for easier computation
        distances_tensor = torch.stack(class_avg_distances)
        
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        q1, q3 = torch.quantile(distances_tensor, torch.tensor([0.25, 0.75], device=distances_tensor.device))
        
        # Calculate IQR (Interquartile Range)
        iqr = q3 - q1
        
        # Define bounds for outlier detection (typically 1.5 * IQR)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        print(f"Lower bound: {lower_bound.item()}, Upper bound: {upper_bound.item()}")

        # Filter out outliers
        valid_distances = distances_tensor[(distances_tensor >= lower_bound) & (distances_tensor <= upper_bound)]
        
        # Calculate mean of non-outlier distances
        if len(valid_distances) > 0:
            avg_distance = torch.mean(valid_distances)
        else:
            # If all are outliers (rare case), just use median
            avg_distance = torch.median(distances_tensor)

        return avg_distance
    def get_query_distance_between_support(self, input, target):
        """
        Compute distance between support samples and query samples
        """

        _, support_idxs, query_idxs = self.get_support_query_idxs(target)
        # Get the support samples
        support_samples = torch.cat([input[idx_list] for idx_list in support_idxs])
        # Get the query samples
        query_samples = input[query_idxs]
        # Compute the pairwise distances
        dists = self.distance_metric.compute(query_samples, support_samples)
        return dists

    def get_openset_distance_between_support(self, input, target, openset):
        """
        Compute distance between support samples and open set samples
        """
        _, support_idxs, query_idxs = self.get_support_query_idxs(target)
        # Get the support samples
        support_samples = torch.cat([input[idx_list] for idx_list in support_idxs])
        # Compute the pairwise distances
        dists = self.distance_metric.compute(openset, support_samples)
        return dists

    def roc_area_calc(self, dist, closed, descending=True):
        """
        計算ROC曲線下面積
        
        參數:
            dist (torch.Tensor): 分數或距離，用於識別開放集樣本
            closed (torch.Tensor): 標記樣本是closed-set(1)還是open-set(-1)的二元標籤
            descending (bool): 排序順序，True表示降序排列（高分表示closed-set）
        
        返回:
            float: ROC曲線下面積
        """
        _, p = dist.sort(descending=descending)
        closed_p = closed[p]
        
        # 計算closed-set和open-set樣本的總數
        total_closed = (closed == 1).sum().item()
        total_open = (closed == -1).sum().item()
        
        if total_closed == 0 or total_open == 0:
            return 0.0
            
        height = 0.0
        width = 0.0
        area = 0.0
        pre = 0  # (0: width; 1: height)
        
        for i in range(len(closed_p)):
            if closed_p[i] == -1:  # open-set樣本
                if pre == 0:
                    area += height * width
                    width = 0.0
                    height += 1.0
                    pre = 1
                else:
                    height += 1.0
            else:  # closed-set樣本
                pre = 0
                width += 1.0
                
        if pre == 0:
            area += height * width
            
        # 除以總高度和總寬度進行標準化
        area = area / total_open / total_closed
        return area

class ProtoLoss(Loss):
    """Prototypical Networks Loss with multiple distance metrics"""
    
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        self.device = opt["settings"]["train"]["device"]

        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
        
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                           f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")

        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                           f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        self.opensetLoss = EntropyLoss()
        
        super().__init__(self.n_support)

    def __call__(self, input, target, opensetTesting=False):
        """
        計算Prototypical Networks的損失值和準確率
        
        Args:
            input: 模型輸出的特徵向量
            target: 目標類別標籤
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        # 移到CPU進行計算（可根據需要修改）
        target_cpu = target.cpu()
        input_cpu = input.cpu()

        if opensetTesting: # openset testing
            num = len(target_cpu) - self.openset_m_samples
            closed_target = target_cpu[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target_cpu))[num:]
            n_query = closed_target.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset: # openset training
            # get support and query indices
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target_cpu, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target_cpu)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = (len(target_cpu) - num_open_samples - self.n_support*n_classes) / n_classes
        else: # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
            n_classes = len(classes)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
        
        # 計算原型
        prototypes = self.compute_prototypes(input_cpu, support_idxs)
        
        # 獲取query樣本
        query_samples = input_cpu[query_idxs]
        # 計算距離
        dists = self.distance_metric.compute(query_samples, prototypes)

        if self.metric_name == 'cosine_similarity':
            # 計算損失和準確率
            loss, acc = self.loss_fn.compute(dists, n_classes, n_query)
        else:
            # 計算損失和準確率
            loss, acc = self.loss_fn.compute(dists, n_classes, n_query, negative_distance=True)
        # openset
        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            openset_samples = input_cpu[openset_idxs]
            openset_dists = self.distance_metric.compute(openset_samples, prototypes)
            
            openset_logits = -openset_dists.view(num_open_samples, n_classes)
            query_logits = -dists.view(n_query*n_classes, n_classes)

            if self.metric_name == 'cosine_similarity':
                openset_logits = -openset_logits
                query_logits = -query_logits

            if self.enable_openset:
                entropy_loss = self.opensetLoss.compute(openset_logits, dim=1)
                open_loss = -entropy_loss

                loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_logits, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(openset_logits, dim=1).max(dim=1)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
             
        return loss, acc

    def get_metric_name(self):
        """返回當前使用的距離度量方法名稱"""
        return self.metric_name

    def get_loss_fn(self):
        """返回當前使用的損失函數名稱"""
        return self.loss_fn_name

class NnLoss(Loss):
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        self.device = opt["settings"]["train"]["device"]

        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
            
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                        f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")
        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                        f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        self.opensetLoss = EntropyLoss()
        
        super().__init__(self.n_support)
    
    def __call__(self, input, target, opensetTesting=False):
        target_cpu = target.cpu()
        input_cpu = input.cpu()

        if opensetTesting: # openset testing
            num = len(target_cpu) - self.openset_m_samples
            closed_target = target_cpu[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target_cpu))[num:]
            n_query = closed_target.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset: # openset training
            # get support and query indices
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target_cpu, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target_cpu)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = (len(target_cpu) - num_open_samples - self.n_support*n_classes) / n_classes
        else: # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
            n_classes = len(classes)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
        
        nn_points = self.get_nn_points(input_cpu, support_idxs)
        
        query_samples = input_cpu[query_idxs]
        
        dists = self.distance_metric.compute(query_samples, nn_points)

        # find the nearest neighbor for each query point
        dists_by_class = dists.view(n_classes*n_query, n_classes, self.n_support)
        min_dists, _ = dists_by_class.min(dim=2)
        
        if self.metric_name == 'cosine_similarity':
            loss, acc = self.loss_fn.compute(min_dists, n_classes, n_query)
        else:
            loss, acc = self.loss_fn.compute(min_dists, n_classes, n_query, negative_distance=True)

        # openset
        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            openset_samples = input_cpu[openset_idxs]
            openset_dists = self.distance_metric.compute(openset_samples, nn_points)

            # find the nearest neighbor for each query point
            openset_dists_by_class = openset_dists.view(num_open_samples, n_classes, self.n_support)
            openset_min_dists, _ = openset_dists_by_class.min(dim=2)

            openset_logits = -openset_min_dists.view(num_open_samples, n_classes)
            query_logits = -min_dists.view(n_query*n_classes, n_classes)

            if self.metric_name == 'cosine_similarity':
                openset_logits = -openset_logits
                query_logits = -query_logits

            if self.enable_openset:
                entropy_loss = self.opensetLoss.compute(openset_logits, dim=1)
                open_loss = -entropy_loss

                loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_logits, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(openset_logits, dim=1).max(dim=1)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
             
        return loss, acc 
    
    def get_metric_name(self):
        return self.metric_name
    
    def get_loss_fn(self):
        return self.loss_fn_name
    
class SoftNnLoss(Loss):
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        self.device = opt["settings"]["train"]["device"]
        
        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
            

        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                        f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")
        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                        f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        self.opensetLoss = EntropyLoss()
        
        super().__init__(self.n_support)
    def __call__(self, input, target, opensetTesting=False):
        """
        Compute soft nearest neighbor loss by using weighted distances to all support samples
        
        Args:
            input: Model output features
            target: Target labels
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        target_cpu = target.cpu()
        input_cpu = input.cpu()       

        if opensetTesting: # openset testing
            num = len(target_cpu) - self.openset_m_samples
            closed_target = target_cpu[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target_cpu))[num:]
            n_query = closed_target.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset: # openset training
            # get support and query indices
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target_cpu, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target_cpu)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = (len(target_cpu) - num_open_samples - self.n_support*n_classes) / n_classes
        else: # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
            n_classes = len(classes)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
         
        
        # Get support points 
        nn_points = self.get_nn_points(input_cpu, support_idxs)
        
        # Get query samples
        query_samples = input_cpu[query_idxs]
        
        # Compute distances between queries and all support samples
        dists = self.distance_metric.compute(query_samples, nn_points)
        
        # Reshape distances to group by class
        dists_by_class = dists.view(n_classes*n_query, n_classes, self.n_support)
        
        # For each query point, compute soft weights for all support samples
        if self.metric_name == 'cosine_similarity':
            weights = F.softmax(dists_by_class, dim=-1)
        else:
            weights = F.softmax(-dists_by_class, dim=-1) 

        weighted_dists = (weights * dists_by_class).sum(dim=2)
            
        # Compute loss and accuracy using weighted distances
        if self.metric_name == 'cosine_similarity':
            loss, acc = self.loss_fn.compute(weighted_dists, n_classes, n_query)
        else:
            loss, acc = self.loss_fn.compute(weighted_dists, n_classes, n_query, negative_distance=True)

        # openset
        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            openset_samples = input_cpu[openset_idxs]
            openset_dists = self.distance_metric.compute(openset_samples, nn_points)

            # find the nearest neighbor for each query point
            openset_dists_by_class = openset_dists.view(num_open_samples, n_classes, self.n_support)
            if self.metric_name == 'cosine_similarity':
                openset_weights = F.softmax(openset_dists_by_class, dim=-1)
            else:
                openset_weights = F.softmax(-openset_dists_by_class, dim=-1)

            openset_weighted_dists = (openset_weights * openset_dists_by_class).sum(dim=2)

            openset_logits = -openset_weighted_dists.view(num_open_samples, n_classes)
            query_logits = -weighted_dists.view(n_query*n_classes, n_classes)

            if self.metric_name == 'cosine_similarity':
                openset_logits = -openset_logits
                query_logits = -query_logits
            if self.enable_openset:
                entropy_loss = self.opensetLoss.compute(openset_logits, dim=1)
                open_loss = -entropy_loss

                loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_logits, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(openset_logits, dim=1).max(dim=1)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
             
        return loss, acc 
class MatchLoss(Loss):
    """Matching Networks Loss with attention-based similarity matching"""
    
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"] 
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        self.device = opt["settings"]["train"]["device"]
        
        # Matching Networks specific parameters
        self.fce = opt.get("settings", {}).get("few_shot", {}).get("parameters").get("fce", False)

        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
        
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                           f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")

        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                           f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        self.opensetLoss = EntropyLoss()

        # 初始化LSTM（如果需要FCE）
        if self.fce:
            None
        super().__init__(self.n_support)


    def apply_fce(self, embeddings):
        """
        應用Full Context Embeddings (FCE) - 使用雙向LSTM處理嵌入向量
        
        Args:
            embeddings: 輸入嵌入向量 [sequence_length, batch_size, feature_dim]
            
        Returns:
            enhanced_embeddings: FCE增強後的嵌入向量
        """
        if not self.fce:
            return embeddings
            
        # 創建雙向LSTM
        hidden_size = embeddings.size(-1) // 2  # 雙向所以除以2
        lstm = nn.LSTM(embeddings.size(-1), hidden_size, 
                      batch_first=False, bidirectional=True)
        
        # 應用LSTM並添加殘差連接
        lstm_out, _ = lstm(embeddings)
        
        # 殘差連接: output = LSTM(input) + input
        enhanced_embeddings = lstm_out + embeddings
        
        return enhanced_embeddings

    def compute_attention_weights(self, query_samples, support_samples):
        """
        計算注意力權重（基於cosine similarity的attention mechanism）
        支援Full Context Embeddings (FCE)
        
        Args:
            query_samples: Query樣本特徵 [n_query, feature_dim]
            support_samples: Support樣本特徵 [n_support, feature_dim]
            
        Returns:
            attention_weights: 注意力權重 [n_query, n_support]
        """
        # 如果啟用FCE，先對embeddings進行增強
        if self.fce:
            # 重新組織為sequence format進行FCE處理
            all_samples = torch.cat([support_samples, query_samples], dim=0)
            all_samples = all_samples.unsqueeze(1)  # [seq_len, 1, feature_dim]
            
            # 應用FCE
            enhanced_samples = self.apply_fce(all_samples)
            enhanced_samples = enhanced_samples.squeeze(1)  # [seq_len, feature_dim]
            
            # 分離support和query
            n_support = support_samples.size(0)
            support_samples = enhanced_samples[:n_support]
            query_samples = enhanced_samples[n_support:]
        
        # 計算cosine similarity作為attention kernel
        similarities = CosineSimilarity.compute(query_samples, support_samples)
        
        # 使用softmax轉換為注意力權重
        attention_weights = F.softmax(similarities, dim=1)
        
        return attention_weights

    def compute_weighted_predictions(self, attention_weights, support_labels, n_classes):
        """
        根據注意力權重計算加權預測
        
        Args:
            attention_weights: 注意力權重 [n_query, n_support]
            support_labels: Support樣本的one-hot標籤 [n_support, n_classes]
            n_classes: 類別數量
            
        Returns:
            predictions: 加權預測 [n_query, n_classes]
        """
        # 計算加權組合: ŷ = Σᵢ a(q, s_i) * y_i
        predictions = torch.mm(attention_weights, support_labels)
        
        return predictions

    def create_one_hot_labels(self, target, support_idxs, n_classes):
        """
        為support樣本創建one-hot標籤
        
        Args:
            target: 目標標籤
            support_idxs: Support樣本索引列表
            n_classes: 類別數量
            
        Returns:
            support_labels: Support樣本的one-hot標籤 [n_support_total, n_classes]
        """
        support_labels_list = []
        
        for class_idx, idx_list in enumerate(support_idxs):
            class_labels = torch.zeros(len(idx_list), n_classes)
            class_labels[:, class_idx] = 1.0
            support_labels_list.append(class_labels)
        
        support_labels = torch.cat(support_labels_list, dim=0)
        return support_labels

    def __call__(self, input, target, opensetTesting=False):
        """
        計算Matching Networks的損失值和準確率
        
        Args:
            input: 模型輸出的特徵向量
            target: 目標類別標籤
            opensetTesting: 是否為開集測試
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        # 移到CPU進行計算
        target_cpu = target.cpu()
        input_cpu = input.cpu()

        if opensetTesting:  # openset testing
            num = len(target_cpu) - self.openset_m_samples
            closed_target = target_cpu[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target_cpu))[num:]
            n_query = closed_target.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset:  # openset training
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(
                target_cpu, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target_cpu)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = (len(target_cpu) - num_open_samples - self.n_support*n_classes) / n_classes
        else:  # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
            n_classes = len(classes)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
        
        # 獲取support和query樣本
        support_samples = self.get_nn_points(input_cpu, support_idxs)
        query_samples = input_cpu[query_idxs]
        
        # 創建support樣本的one-hot標籤
        support_labels = self.create_one_hot_labels(target_cpu, support_idxs, n_classes)
        
        # 計算注意力權重
        attention_weights = self.compute_attention_weights(query_samples, support_samples)
        
        # 計算加權預測
        weighted_predictions = self.compute_weighted_predictions(attention_weights, support_labels, n_classes)
        
        # 準備目標標籤
        query_targets = []
        for class_idx, class_label in enumerate(classes):
            query_targets.extend([class_idx] * n_query)
        query_targets = torch.tensor(query_targets, dtype=torch.long)

        # 計算交叉熵損失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(weighted_predictions, query_targets)
        
        # 計算準確率
        _, predicted = weighted_predictions.max(1)
        acc = predicted.eq(query_targets).float().mean()
        
        # 處理開集情況
        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            openset_samples = input_cpu[openset_idxs]
            
            # 計算開集樣本的注意力權重和預測
            openset_attention_weights = self.compute_attention_weights(openset_samples, support_samples)
            openset_predictions = self.compute_weighted_predictions(openset_attention_weights, support_labels, n_classes)
            
            # 計算熵損失來鼓勵開集樣本的不確定性
            if self.enable_openset:
                entropy_loss = self.opensetLoss.compute(openset_predictions, dim=1)
                open_loss = -entropy_loss  # 負熵損失，鼓勵高熵
                
                loss = loss + self.openset_loss_scale * open_loss
            
            # 計算AUROC
            query_max_probs = F.softmax(weighted_predictions, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(openset_predictions, dim=1).max(dim=1)[0]
            
            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)
            
            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
             
        return loss, acc

    def get_metric_name(self):
        """返回當前使用的距離度量方法名稱"""
        return self.metric_name

    def get_loss_fn(self):
        """返回當前使用的損失函數名稱"""
        return self.loss_fn_name

    def get_attention_info(self):
        """返回注意力機制相關資訊"""
        return {
            "fce": self.fce,
            "attention_layers": self.attention_layers
        }

class RelationNetwork(nn.Module, Loss):
    """
    Few-shot Relation Network，採 GCN 做 relation learning。
    需傳入外部 encoder（CNN / Transformer / …），
    encoder 產生節點特徵後交由 GCN 傳遞。
    """
    def __init__(self, opt: dict, encoder: nn.Module):
        super(RelationNetwork, self).__init__()
        self.opt = opt
        self.device = opt["settings"]["train"]["device"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)

        self.encoder = encoder
        self.relation = MLPRelationModule(2 * self.opt["settings"]["model"]["output_size"])

        self.ce = nn.CrossEntropyLoss()
        self.opensetLoss = EntropyLoss()

    def forward(self, data, opensetTesting: bool = False):

        input = self.encoder(data)
        target = data.y

        if opensetTesting: # openset testing
            num = len(target) - self.openset_m_samples
            closed_target = target[:num]
            _, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            num_support_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target))[num:]
        elif self.enable_openset: # openset training
            # get support and query indices
            _, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target, openset=True, cls_openset=self.cls_training_openset)
            num_support_classes = len(torch.unique(target)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
        else: # closed set training & testing
            _, support_idxs, query_idxs = self.get_support_query_idxs(target)
            num_support_classes = len(torch.unique(target))
            num_open_samples = 0

        num_queries = (20 - self.n_support) * num_support_classes  # 每個類別20個query樣本

        s_labels_ori = torch.cat([target[idx_list] for idx_list in support_idxs])
        q_labels_ori = target[query_idxs]
        unique_labels = torch.unique(torch.cat([s_labels_ori, q_labels_ori]))
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        s_labels_mapped = torch.tensor([label_map[label.item()] for label in s_labels_ori]).to(self.device)
        q_labels_mapped = torch.tensor([label_map[label.item()] for label in q_labels_ori]).to(self.device)
        # generate one-hot labels
        s_labels = F.one_hot(s_labels_mapped, num_support_classes)
        q_labels = F.one_hot(q_labels_mapped, num_support_classes)

        prototypes = self.compute_prototypes(input, support_idxs)
        query_samples = input[query_idxs]

        # 為了處理 openset 情況，如果有 openset samples，也將它們加入 query
        if self.enable_openset or opensetTesting:
            if num_open_samples > 0:
                openset_samples = input[openset_idxs]  # shape: [num_open_samples, 128]
                query_samples = torch.cat([query_samples, openset_samples], dim=0)

        prototypes_ext = prototypes.unsqueeze(0).repeat(num_queries + num_open_samples, 1, 1) # [num_queries, num_support_classes, 128]
        query_samples_ext = query_samples.unsqueeze(1).repeat(1, num_support_classes, 1)  # [num_queries, num_support_classes, 128]
        relation_pairs = torch.cat([prototypes_ext, query_samples_ext], dim=2)  # [num_queries, num_support_classes, 256]
        relation_pairs = relation_pairs.view(-1, 2 * self.opt["settings"]["model"]["output_size"])
        # 通過 relation network 計算 relation scores
        self.relation.to(self.device)
        relations = self.relation(relation_pairs)  # [num_queries, 1]
        relationscore = relations.view(num_queries + num_open_samples, num_support_classes)
        # 計算損失和準確率
        query_relationscore = relationscore[:num_queries, :]  # [num_queries , num_support_classes]
        gtq = torch.argmax(q_labels, 1)
        loss = self.ce(query_relationscore, gtq)
        # 計算準確率
        predq = torch.argmax(query_relationscore, 1)
        correct = (predq == gtq).sum()

        total = num_queries
        acc = 1.0 * correct.float() / float(total)

        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            if self.enable_openset:
                entropy_loss = self.opensetLoss.compute(relationscore[num_queries:], dim=1)
                open_loss = -entropy_loss 
                loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_relationscore, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(relationscore[num_queries:], dim=1).max(dim=1)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)

        torch.cuda.empty_cache()
        return loss, acc

class MAMLLoss(nn.Module, Loss):
    """MAML Loss for Few-Shot Learning"""
    
    def __init__(self, opt: dict, encoder: nn.Module):
        super(MAMLLoss, self).__init__()
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        self.device = opt["settings"]["train"]["device"]

        self.first_order = opt.get("settings", {}).get("few_shot", {}).get("parameters", {}).get("first_order", False)
        # Gradient clipping
        self.grad_clip = opt.get("settings", {}).get("few_shot", {}).get("parameters", {}).get("grad_clip", None)

        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
        
        self.loss_functions = {
            'CrossEntropyLoss': F.cross_entropy,
            'MSE': F.mse_loss,
            'smooth_l1': F.smooth_l1_loss
        }
        
        if self.loss_fn_name not in self.loss_functions:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                           f"Supported loss functions are: {list(self.loss_functions.keys())}")
        
        self.loss_fn = self.loss_functions[self.loss_fn_name]

        self.opensetLoss = EntropyLoss()

        self.encoder = encoder
    
        
    def __call__(self, data, opensetTesting=False):

        input = data
        target = data.y

        if opensetTesting:  # openset testing
            num = len(target) - self.openset_m_samples
            closed_target = target[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target))[num:]
            n_query = closed_target.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset:  # openset training
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(
                target, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = (len(target) - num_open_samples - self.n_support * n_classes) / n_classes
        else:  # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target)
            n_classes = len(classes)
            n_query = target.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
        # 原本: support_samples = torch.cat([input[idx_list] for idx_list in support_idxs])
        support_graphs = []
        for idx_list in support_idxs:
            support_graphs.extend(self.extract_graphs_from_batch(input, idx_list))
        support_samples = Batch.from_data_list(support_graphs)

        # 原本: query_samples = input[query_idxs]
        query_graphs = self.extract_graphs_from_batch(input, query_idxs)
        query_samples = Batch.from_data_list(query_graphs)

        # get support labels
        support_targets = []
        for class_idx, class_label in enumerate(classes):
            support_targets.extend([class_idx] * self.n_support)
        support_targets = torch.tensor(support_targets, dtype=torch.long).to(self.device)
        # 準備目標標籤
        query_targets = []
        for class_idx, class_label in enumerate(classes):
            query_targets.extend([class_idx] * n_query)
        query_targets = torch.tensor(query_targets, dtype=torch.long).to(self.device)
        
        is_no_grad_mode = not torch.is_grad_enabled()
        if is_no_grad_mode:
            with torch.enable_grad():
                self.encoder.train()
                adapted_model, inner_losses = self.inner_loop_update(self.encoder, support_samples, support_targets)
        else:
            adapted_model, inner_losses = self.inner_loop_update(self.encoder, support_samples, support_targets)

        if is_no_grad_mode:
            adapted_model.eval()
        query_logits = adapted_model(query_samples)
        meta_loss = self.loss_fn(query_logits, query_targets)

        query_prob = F.softmax(query_logits, dim=1)
        predq = query_prob.argmax(dim=1)
        correct = (predq == query_targets).float().sum()
        acc = correct / len(query_targets)

        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            openset_graphs = self.extract_graphs_from_batch(input, openset_idxs)
            openset_samples = Batch.from_data_list(openset_graphs)
            openset_logits = adapted_model(openset_samples)

            if self.enable_openset:
                entropy_loss = self.opensetLoss.compute(openset_logits, dim=1)
                open_loss = -entropy_loss 
                meta_loss = meta_loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_logits, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(openset_logits, dim=1).max(dim=1)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)

        torch.cuda.empty_cache()

        return meta_loss, acc

    def inner_loop_update(self, model, support_input, support_target):
        """執行inner loop的梯度更新"""
        # 創建模型的副本用於更新
        adapted_model = self.create_adapted_model(model)
        inner_losses = []
        
        for step in range(10):
            # 前向傳播
            support_output = adapted_model(support_input)
            inner_loss = self.loss_fn(support_output, support_target)
            inner_losses.append(inner_loss.item())
            
            # 計算梯度
            if self.first_order:
                # 一階近似：不計算二階梯度
                grads = torch.autograd.grad(
                    inner_loss, adapted_model.parameters(), 
                    create_graph=False, retain_graph=False
                )
            else:
                # 二階梯度：保持計算圖
                grads = torch.autograd.grad(
                    inner_loss, adapted_model.parameters(), 
                    create_graph=True, retain_graph=True
                )
            
            # 梯度裁剪
            if self.grad_clip is not None:
                grads = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grads]
            
            # 更新參數
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - 0.05 * grad
        
        return adapted_model, inner_losses
    
    def create_adapted_model(self, model):
        """創建一個可以進行內層更新的模型副本"""
        # 深度複製模型
        adapted_model = copy.deepcopy(model)
        
        # 確保所有參數都需要梯度
        for param in adapted_model.parameters():
            param.requires_grad = True
            
        return adapted_model
    
    def extract_graphs_from_batch(self, data_batch, graph_indices):
        """從DataBatch中提取指定索引的圖"""
        graphs = []
        
        for graph_idx in graph_indices:
            # 找到屬於這個圖的節點
            node_mask = data_batch.batch == graph_idx
            node_indices = torch.where(node_mask)[0]
            
            # 提取節點特徵
            x = data_batch.x[node_mask]
            
            # 提取邊索引並重新編號
            edge_mask = node_mask[data_batch.edge_index[0]] & node_mask[data_batch.edge_index[1]]
            edge_index = data_batch.edge_index[:, edge_mask]
            
            # 重新編號節點索引 (從0開始)
            node_mapping = torch.zeros(data_batch.x.size(0), dtype=torch.long, device=data_batch.x.device)
            node_mapping[node_indices] = torch.arange(len(node_indices), device=data_batch.x.device)
            edge_index = node_mapping[edge_index]
            
            # 創建單個圖的Data物件
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                y=data_batch.y[graph_idx] if hasattr(data_batch, 'y') else None,
                label=data_batch.label[graph_idx] if hasattr(data_batch, 'label') else None
            )
            graphs.append(graph_data)
        
        return graphs

    def get_inner_lr(self):
        """返回內層學習率"""
        return self.inner_lr
    
    def get_num_inner_updates(self):
        """返回內層更新步數"""
        return self.num_inner_updates
    
    def set_inner_lr(self, new_lr):
        """設置新的內層學習率"""
        self.inner_lr = new_lr
    
    def get_loss_fn_name(self):
        """返回當前使用的損失函數名稱"""
        return self.loss_fn_name
class ClassifierLoss(Loss):
    def __init__(self, opt: dict):
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn = opt["settings"]["train"]["loss"]

        if self.loss_fn == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_fn == "MSELoss":
            self.criterion = nn.MSELoss()
        elif self.loss_fn == "NLLLoss":
            self.criterion = nn.NLLLoss()
        
        super().__init__(self.n_support)
        
    def __call__(self, model, input, target):
        target_cpu = target.cpu()
        input_cpu = input.cpu()
        
        classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
        n_classes = len(classes)
        n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        
        # get support values
        support_samples = input_cpu[support_idxs]
        
        loss = self.criterion(support_samples, target_cpu[support_idxs])
        acc = torch.sum(torch.argmax(support_samples, dim=1) == target_cpu[support_idxs]).item() / len(support_idxs)
        
        return loss, acc

from torch_geometric.data import Batch
class LabelPropagation(nn.Module, Loss):
    """Label Propagation"""
    def __init__(self, opt: dict, encoder):
        super(LabelPropagation, self).__init__()
        self.opt = opt
        self.args = opt['settings']['few_shot']['parameters']
        self.device = opt["settings"]["train"]["device"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
        self.relation_model = opt.get("settings", {}).get("few_shot", {}).get("parameters", {}).get("relation_model", "GraphSAGE")
        self.encoder = encoder
        self.relation = GraphRelationNetwork(self.args["dim_in"], self.args["dim_hidden"], self.args["dim_out"], self.args["relation_layer"], self.relation_model)

        self.metric_name = opt["settings"]["train"]["distance"]
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.opensetLoss = EntropyLoss()
        
        if opt['settings']['few_shot']['parameters']['rn'] == 300:   # learned sigma, fixed alpha
            self.alpha = torch.tensor([opt['settings']['few_shot']['parameters']['alpha']], requires_grad=False).to(self.device)
        elif opt['settings']['few_shot']['parameters']['rn'] == 30:    # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([opt['settings']['few_shot']['parameters']['alpha']]).to(self.device), requires_grad=True)

    def forward(self, data, opensetTesting=False):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        import torch.nn.functional as F
        eps = np.finfo(float).eps

        input = self.encoder(data.x, data.edge_index)
        target = data.y
        edge_index = data.edge_index
        batch = data.batch

        if opensetTesting: # openset testing
            num = len(target) - self.openset_m_samples
            closed_target = target[:num]
            _, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            num_support_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target))[num:]
        elif self.enable_openset: # openset training
            # get support and query indices
            _, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target, openset=True, cls_openset=self.cls_training_openset)
            num_support_classes = len(torch.unique(target)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
        else: # closed set training & testing
            _, support_idxs, query_idxs = self.get_support_query_idxs(target)
            num_support_classes = len(torch.unique(target))
            num_open_samples = 0
        num_queries = 20 - self.n_support

        s_labels_ori = torch.cat([target[idx_list] for idx_list in support_idxs])
        q_labels_ori = target[query_idxs]
        unique_labels = torch.unique(torch.cat([s_labels_ori, q_labels_ori]))
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        s_labels_mapped = torch.tensor([label_map[label.item()] for label in s_labels_ori]).to(self.device)
        q_labels_mapped = torch.tensor([label_map[label.item()] for label in q_labels_ori]).to(self.device)
        # generate one-hot labels
        s_labels = F.one_hot(s_labels_mapped, num_support_classes)
        q_labels = F.one_hot(q_labels_mapped, num_support_classes)

        pool_input = global_add_pool(input, batch)

        # Step1: Embedding
        N = pool_input.shape[0]

        # Step2: Graph Construction
        ## sigmma
        if self.args['rn'] in [30,300]:
            self.relation.to(self.device)
            #self.sigma = self.relation(input, edge_index, batch)
            self.sigma = F.softplus(self.relation(input, edge_index, batch))
            # sigma_pooled = scatter_mean(self.sigma, batch, dim=0)
            # self.sigma = torch.clamp(self.sigma, min=0.1, max=10.0) -> No need to clamp

            self.sigma_support = torch.cat([self.sigma[idx_list] for idx_list in support_idxs])
            self.sigma_query = self.sigma[query_idxs]

            if self.enable_openset or opensetTesting:
                self.sigma_openset = self.sigma[openset_idxs]
                self.sigma = torch.cat([self.sigma_support, self.sigma_query, self.sigma_openset], 0)
            else:
                self.sigma = torch.cat([ self.sigma_support, self.sigma_query], 0)

            pool_input_support = torch.cat([pool_input[idx_list] for idx_list in support_idxs])
            pool_input_query = pool_input[query_idxs]

            if self.enable_openset or opensetTesting:
                pool_input_openset = pool_input[openset_idxs]
                pool_input = torch.cat([pool_input_support, pool_input_query, pool_input_openset], 0)
            else:
                pool_input = torch.cat([ pool_input_support, pool_input_query], 0)

            pool_input = pool_input / (self.sigma + eps)
            emb1 = torch.unsqueeze(pool_input, 1)
            emb2 = torch.unsqueeze(pool_input, 0)
            W = ((emb1 - emb2) ** 2).mean(2)
            W = torch.exp(-W / 2)

        ## keep top-k values
        if self.args['k'] > 0:
            topk, indices = torch.topk(W, self.args['k'])
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
            W = W * mask
        # 正規化
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(num_support_classes * num_queries, num_support_classes).to(self.device)

        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            yo = torch.zeros(num_open_samples, num_support_classes).to(self.device)
            y = torch.cat((ys, yu, yo), 0)
        else:
            y = torch.cat((ys, yu), 0)
        
        F_ = torch.matmul(torch.inverse(torch.eye(N).to(self.device) - self.alpha * S + eps), y)

        # Fq = F_[num_support_classes * self.n_support:, :]
        Fq = F_[num_support_classes * self.n_support : num_support_classes * self.n_support + num_support_classes * num_queries, :]

        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            Fo = F_[num_support_classes * self.n_support + num_support_classes * num_queries:, :]

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().to(self.device)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        F_sq = F_[:num_support_classes * self.n_support + num_support_classes * num_queries, :]

        loss = ce(F_sq, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()

        total = num_queries * num_support_classes
        acc = 1.0 * correct.float() / float(total)

        if (self.enable_openset or opensetTesting) and num_open_samples > 0:
            # calculate open set loss
            if self.enable_openset:
                entropy_loss = self.opensetLoss.compute(Fo, dim=1)
                open_loss = -entropy_loss
                loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(Fq, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(Fo, dim=1).max(dim=1)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
        
        torch.cuda.empty_cache()
        
        return loss, acc
    
    def get_encoded_data(self, data):
        """
        Encode input data using the model and return the encoded results
        """
        input = self.encoder(data.x, data.edge_index)
        batch = data.batch

        from torch_geometric.nn import global_add_pool
        pool_input = global_add_pool(input, batch)

        return pool_input

    def get_processed_data(self, data):
            """
            Process data through the model and return the processed results
            
            Parameters:
            data: The input data
            
            Returns:
            The processed data (N_way*N_shot + N_way*N_query) x N_features
            """
            import torch.nn.functional as F
            eps = np.finfo(float).eps

            # Extract initial embeddings
            input = self.encoder(data.x, data.edge_index)
            target = data.y
            edge_index = data.edge_index
            batch = data.batch

            # Get support and query indices
            _, support_idxs, query_idxs = self.get_support_query_idxs(target)
            num_classes = len(torch.unique(target))
            num_support = self.opt["settings"]["few_shot"]["train"]["support_shots"]
            num_queries = 20 - num_support

            # Process labels
            s_labels_ori = torch.cat([target[idx_list] for idx_list in support_idxs])
            q_labels_ori = target[query_idxs]
            unique_labels = torch.unique(torch.cat([s_labels_ori, q_labels_ori]))
            label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
            s_labels_mapped = torch.tensor([label_map[label.item()] for label in s_labels_ori]).to(self.device)
            q_labels_mapped = torch.tensor([label_map[label.item()] for label in q_labels_ori]).to(self.device)
            
            # Generate one-hot labels
            s_labels = F.one_hot(s_labels_mapped, num_classes)
            q_labels = F.one_hot(q_labels_mapped, num_classes)

            # Pool embeddings
            pool_input = global_add_pool(input, batch)

            # Graph construction
            if self.args['rn'] in [30, 300]:
                self.relation.to(self.device)
                sigma = F.softplus(self.relation(input, edge_index, batch))
                
                sigma_support = torch.cat([sigma[idx_list] for idx_list in support_idxs])
                sigma_query = sigma[query_idxs]
                sigma_combined = torch.cat([sigma_support, sigma_query], 0)
                
                pool_input_support = torch.cat([pool_input[idx_list] for idx_list in support_idxs])
                pool_input_query = pool_input[query_idxs]
                pool_input_combined = torch.cat([pool_input_support, pool_input_query], 0)
                
                normalized_pool_input = pool_input_combined / (sigma_combined + eps)
            
            return normalized_pool_input