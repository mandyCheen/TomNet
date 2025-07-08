import pickle
import networkx as nx
import pandas as pd
import os
import json
from multiprocessing import Pool, Manager
from tqdm import tqdm
import functools

DATA_FOLDER = "/home/manying/Data/malware/Malware202403/fcg_ghidra/X86_64"
OUTPUT_FOLDER = "/home/manying/Projects/fcgFewShot/dataset/data_ghidra_fcg_openset"
CSV_PATH = "/home/manying/Projects/fcgFewShot/dataset/raw_csv/malware_diec_ghidra_x86_64_fcg_openset_dataset.csv"
ERROR_FILE = "/home/manying/Projects/fcgFewShot/openset/genGpickle_openset_error.txt"

def process_single_file(file_info, error_queue):
    """處理單個文件的函數"""
    try:
        file_name, cpu, family = file_info
        
        # 建立輸出路徑
        output_folder = f"{OUTPUT_FOLDER}/{cpu}/{family}"
        output_path = f"{output_folder}/{file_name}.gpickle"
        if os.path.exists(output_path):
            return True
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # 讀取並處理檔案
        dot_file = f"{DATA_FOLDER}/{file_name}/{file_name}.dot"
        json_file = f"{DATA_FOLDER}/{file_name}/{file_name}.json"
        
        G = nx.drawing.nx_pydot.read_dot(dot_file)
        with open(json_file, "r") as f:
            data = json.load(f)
            
        for node in data.keys():
            instructions = data[node]['instructions']
            opcode = []
            for ins in instructions:
                opcode.append(ins['instruction'].split()[0])
            G.nodes[node]['x'] = opcode
            
        with open(output_path, "wb") as f:
            pickle.dump(G, f)
            
        return True
        
    except Exception as e:
        error_message = f"{file_name}: {str(e)}\n"
        error_queue.put(error_message)
        return False

def write_errors(error_queue):
    """將錯誤訊息寫入檔案"""
    with open(ERROR_FILE, "a") as f:
        while not error_queue.empty():
            error_message = error_queue.get()
            f.write(error_message)

def main():
    # 讀取數據集
    dataset = pd.read_csv(CSV_PATH)
    
    # 準備處理數據
    file_info_list = [(row["file_name"], row["CPU"], row["family"]) 
                      for _, row in dataset.iterrows()]
    
    # 設置進程池和共享隊列
    num_processes = 20 # os.cpu_count()  # 使用可用的CPU核心數
    manager = Manager()
    error_queue = manager.Queue()
    
    # 建立處理函數的偏函數，包含error_queue
    process_func = functools.partial(process_single_file, error_queue=error_queue)
    
    # 使用進程池處理檔案，並顯示進度條
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, file_info_list),
            total=len(file_info_list),
            desc="Processing files",
            unit="file"
        ))
    
    # 處理錯誤訊息
    write_errors(error_queue)
    
    # 輸出處理結果統計
    successful = sum(results)
    total = len(results)
    print(f"\nProcessing completed:")
    print(f"- Successfully processed: {successful}/{total} files")
    print(f"- Failed: {total-successful} files")
    print(f"Error details have been written to: {ERROR_FILE}")

if __name__ == "__main__":
    main()