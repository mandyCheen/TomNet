import argparse
from utils import load_config
from loadDataset import LoadDataset
from trainModule import TestModule
from fcgVectorize import FCGVectorize
import os
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Few Shot FCG")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    configPath = os.path.abspath(args.config)
    options = load_config(configPath)
    
    dataset = LoadDataset(options)

    errorPath = os.path.dirname(configPath) + "/testError.txt"
    try:
        test = TestModule(configPath, dataset)
        test.eval()
    except Exception as e:
        with open(errorPath, "w") as f:
            f.write(str(e))
        raise