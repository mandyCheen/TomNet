import argparse
from utils import load_config
from loadDataset import LoadDataset
from trainModule import TrainModule, TestModule
from fcgVectorize import FCGVectorize
import warnings
import os
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Few Shot FCG")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    options = load_config(args.config)
    
    dataset = LoadDataset(options)
    
    vectorize = FCGVectorize(options, dataset)
    vectorize.node_embedding(dataset.rawDataset)
    
    train = TrainModule(options, dataset)
    errorPath = train.model_folder + "/error.txt"
    try:
        train.train()

        test = TestModule(os.path.join(train.model_folder, "config.json"), dataset)
        test.eval()
    except Exception as e:
        with open(errorPath, "w") as f:
            f.write(str(e))
        raise


          