import warnings, sys, os
import multiprocessing as mp
from itertools import product
import torch
import gc
import resource
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, save_config
from loadDataset import LoadDataset
from fcgVectorize import FCGVectorize
from trainModule import TrainModule
from trainModule import TestModule

warnings.filterwarnings("ignore")

# 設置文件描述符限制
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
except:
    pass

def setup_logging(process_id=None):
    """
    設置日誌記錄
    """
    # 創建logs目錄
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成時間戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if process_id is not None:
        log_filename = f"{log_dir}/process_{process_id}_{timestamp}.log"
        logger_name = f"process_{process_id}"
    else:
        log_filename = f"{log_dir}/main_{timestamp}.log"
        logger_name = "main"
    
    # 創建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 清除已存在的handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 創建文件handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 創建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    
    return logger

def worker_process(process_id, task_batch, cuda_device):
    """
    工作進程函數
    Args:
        process_id: 進程ID
        task_batch: 分配給此進程的任務批次
        cuda_device: CUDA設備編號
    """
    # 設置進程專用的logger
    logger = setup_logging(process_id)
    
    logger.info(f"Process {process_id} started on {cuda_device}")
    
    # 設置PyTorch多進程共享策略
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # 設置CUDA設備
    torch.cuda.set_device(cuda_device)
    
    # 初始化進程內的變量
    dataset = None
    
    for i, (seed, exp, lambda_) in enumerate(task_batch):
        try:
            logger.info(f"Process {process_id} - Task {i+1}/{len(task_batch)} - seed: {seed}, exp: {exp}, lambda: {lambda_}")
            
            # 清理之前的資源
            if 'trainModule' in locals():
                del trainModule
            if 'test' in locals():
                del test
            gc.collect()
            torch.cuda.empty_cache()
            
            # 載入配置
            options = load_config("../config/config_label_prop_openset_meta_nict.json")
            
            # 設置實驗參數
            options["settings"]["name"] = exp + f"_LabelPropagation_alpha0.7_k20_gcn_lambda{lambda_}_proc{process_id}"
            shots = int(exp.split("_")[1].split("shot")[0])
            way = int(exp.split("_")[0].split("way")[0])
            
            options["settings"]["few_shot"]["train"]["support_shots"] = shots
            options["settings"]["few_shot"]["train"]["query_shots"] = 20 - shots
            options["settings"]["few_shot"]["test"]["support_shots"] = shots
            options["settings"]["few_shot"]["test"]["query_shots"] = 20 - shots
            options["settings"]["few_shot"]["train"]["class_per_iter"] = way
            options["settings"]["few_shot"]["test"]["class_per_iter"] = way
            
            # 設置CUDA設備
            options["settings"]["train"]["device"] = cuda_device
            options["settings"]["openset"]["train"]["loss_weight"] = lambda_
            options["settings"]["seed"] = seed
            
            # 為每個進程創建獨立的配置文件
            config_path = f"../config/config_label_prop_openset_meta_nict_proc{process_id}.json"
            save_config(options, config_path)
            
            # 只在第一次或需要時載入數據集
            if dataset is None or i == 0:
                if dataset is not None:
                    del dataset
                    gc.collect()
                logger.info(f"Process {process_id} - Loading dataset")
                dataset = LoadDataset(options)
            
            # 訓練
            logger.info(f"Process {process_id} - Starting training")
            trainModule = TrainModule(options, dataset)
            trainModule.train()
            
            # 測試
            logger.info(f"Process {process_id} - Starting testing")
            test = TestModule(os.path.join(trainModule.model_folder, "config.json"), dataset)
            test.eval()
            
            # 立即清理資源
            del trainModule
            del test
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"Process {process_id} - Task {i+1}/{len(task_batch)} completed successfully")
            
        except Exception as e:
            logger.error(f"Process {process_id} error with task (seed={seed}, exp={exp}, lambda={lambda_}): {str(e)}")
            logger.error(f"Process {process_id} - Exception details: {type(e).__name__}: {e}")
            # 強制清理資源
            gc.collect()
            torch.cuda.empty_cache()
            continue
    
    # 最終清理
    if dataset is not None:
        del dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info(f"Process {process_id} completed all tasks")
    
    # 關閉logger handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def distribute_tasks(tasks, num_processes):
    """
    將任務平均分配給各個進程
    """
    tasks_per_process = len(tasks) // num_processes
    remainder = len(tasks) % num_processes
    
    distributed_tasks = []
    start_idx = 0
    
    for i in range(num_processes):
        # 如果有餘數，前面的進程多分配一個任務
        end_idx = start_idx + tasks_per_process + (1 if i < remainder else 0)
        distributed_tasks.append(tasks[start_idx:end_idx])
        start_idx = end_idx
    
    return distributed_tasks

def main():
    # 設置主進程的logger
    main_logger = setup_logging()
    
    # 實驗設置
    expList = ["5way_5shot", "5way_10shot", "10way_5shot", "10way_10shot"]
    seeds = [6, 22, 31, 42, 888, 7, 10, 666, 11, 19]
    lambdaList = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 創建所有任務的組合
    all_tasks = list(product(seeds, expList, lambdaList))
    main_logger.info(f"Total tasks: {len(all_tasks)}")
    
    # 設置進程數和CUDA設備
    num_processes = 8
    cuda_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    
    # 檢查CUDA設備可用性
    available_devices = []
    for device in cuda_devices:
        try:
            torch.cuda.set_device(device)
            available_devices.append(device)
            main_logger.info(f"Device {device} is available")
        except Exception as e:
            main_logger.warning(f"Device {device} is not available: {e}")
    
    if len(available_devices) < num_processes:
        main_logger.warning(f"Only {len(available_devices)} CUDA devices available, but {num_processes} processes requested")
        num_processes = len(available_devices)
        cuda_devices = available_devices
    
    # 分配任務
    task_batches = distribute_tasks(all_tasks, num_processes)
    
    # 顯示任務分配信息
    for i, batch in enumerate(task_batches):
        main_logger.info(f"Process {i} assigned {len(batch)} tasks on {cuda_devices[i]}")
    
    # 設置multiprocessing context
    ctx = mp.get_context('spawn')
    
    # 創建並啟動進程
    processes = []
    
    try:
        main_logger.info("Starting all processes...")
        for i in range(num_processes):
            p = ctx.Process(
                target=worker_process,
                args=(i, task_batches[i], cuda_devices[i])
            )
            p.start()
            processes.append(p)
            main_logger.info(f"Process {i} started")
        
        # 等待所有進程完成
        main_logger.info("Waiting for all processes to complete...")
        for i, p in enumerate(processes):
            p.join()
            main_logger.info(f"Process {i} finished")
        
        main_logger.info("All processes completed successfully!")
        
    except KeyboardInterrupt:
        main_logger.warning("Interrupted by user. Terminating processes...")
        for i, p in enumerate(processes):
            p.terminate()
            p.join()
            main_logger.info(f"Process {i} terminated")
    
    except Exception as e:
        main_logger.error(f"Error occurred: {str(e)}")
        main_logger.error(f"Exception details: {type(e).__name__}: {e}")
        for i, p in enumerate(processes):
            p.terminate()
            p.join()
            main_logger.info(f"Process {i} terminated due to error")
    
    # 關閉main logger handlers
    for handler in main_logger.handlers:
        handler.close()
        main_logger.removeHandler(handler)

if __name__ == "__main__":
    # 設置multiprocessing的啟動方法和共享策略
    mp.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()