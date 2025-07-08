from loadDataset import LoadDataset
import pandas as pd
from tqdm import tqdm
import pickle
import os   
from gensim.models import Word2Vec
import sklearn.preprocessing as labelEncoder
import torch

class FCGVectorize():
    def __init__(self, opt: dict, dataset: LoadDataset, pretrain: bool = False):
        self.nodeEmbedding = opt["settings"]["vectorize"]["node_embedding_method"]
        if pretrain:
            self.data_root = opt["paths"]["data"]["pretrain_dataset"]
        else:
            self.data_root = opt["paths"]["data"]["fcg_dataset"]
        self.trainDataset = dataset.trainData
        self.testDataset = dataset.testData 
        self.valDataset = dataset.valData
        self.embeddingFolder = os.path.join(opt["paths"]["data"]["embedding_folder"], dataset.datasetName, self.nodeEmbedding)
        self.embeddingSize = opt["settings"]["vectorize"]["node_embedding_size"]
        self.numWorkers = opt["settings"]["vectorize"]["num_workers"]
        if dataset.enable_openset:
            self.opensetData = dataset.opensetData
            self.openset_data_root = opt["paths"]["data"]["openset_dataset"]
        if not os.path.exists(self.embeddingFolder):
            os.makedirs(self.embeddingFolder)

    def node_embedding(self, dataset: pd.DataFrame, openset: bool = False):
        if openset:
            self.data_root = self.openset_data_root
        if self.nodeEmbedding == "word2vec":
            self.opcodeSetPath = os.path.join(self.embeddingFolder, "opcodeSet.pkl")
            self.sentencesPath = os.path.join(self.embeddingFolder, "opcodeSentences.pkl")
            model = self.word2vec() 
        print("Start to get node embedding...")
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            cpu = row["CPU"]
            family = row["family"]
            fileName = row["file_name"]
            filePath = f"{self.data_root}/{cpu}/{family}/{fileName}.gpickle"
            outputPath = f"{self.embeddingFolder}/{cpu}/{family}/{fileName}.gpickle"
            nodes_to_remove = []
            if not os.path.exists(outputPath):
                with open(filePath, "rb") as f:
                    fcg = pickle.load(f)
                    for node in fcg.nodes():
                        avgOpcodeEmbedding = []
                        if "x" not in fcg.nodes[node]:
                            nodes_to_remove += [node]
                            continue
                        opcode = fcg.nodes[node]["x"]
                        opcode = [op for op in opcode if op in model.wv]
                        if opcode == []:
                            avgOpcodeEmbedding = [0] * self.embeddingSize
                            fcg.nodes[node]["x"] = avgOpcodeEmbedding
                            continue
                        opcodeEmbedding = model.wv[opcode]
                        opcodeEmbedding = torch.tensor(opcodeEmbedding)
                        avgOpcodeEmbedding = torch.mean(opcodeEmbedding, dim=0)
                        fcg.nodes[node]["x"] = avgOpcodeEmbedding
                    fcg.remove_nodes_from(nodes_to_remove)
                if not os.path.exists(f"{self.embeddingFolder}/{cpu}/{family}"):
                    os.makedirs(f"{self.embeddingFolder}/{cpu}/{family}")
                with open(outputPath, "wb") as f:
                    pickle.dump(fcg, f)
        print("Finish getting node embedding")
                
    def genOpcodeSet(self, opcodeSet: set, sentences: list, dataset: pd.DataFrame):
        print("Start to get OpcodeSet & Sentence...")
        for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            cpu = row["CPU"]
            family = row["family"]
            fileName = row["file_name"]
            filePath = f"{self.data_root}/{cpu}/{family}/{fileName}.gpickle"
            with open(filePath, "rb") as f:
                fcg = pickle.load(f)
                for node in fcg.nodes():
                    if "x" not in fcg.nodes[node]:
                        continue
                    opcode = fcg.nodes[node]["x"]
                    opcodeSet.update(opcode)
                    sentences.append(opcode)
           
        print(f"Save opcodeSet to {self.opcodeSetPath}")
        with open(self.opcodeSetPath, "wb") as f:
            pickle.dump(opcodeSet, f)
        print(f"Save sentence to {self.sentencesPath}")
        with open(self.sentencesPath, "wb") as f:
            pickle.dump(sentences, f)
        print("Finish getting opcodeSet & Sentence")


    def word2vec(self) -> Word2Vec:
        # Train word2vec model
        if not os.path.exists(os.path.join(self.embeddingFolder, "opcode2vec.model")):
            if not os.path.exists(self.opcodeSetPath):
                print("Training opcodeSet & Sentence not exist, start to get opcodeSet & Sentence...")
                opcodeSet = set()
                sentences = []
                self.genOpcodeSet(opcodeSet, sentences, self.trainDataset)
            else:
                print("Training opcodeSet & Sentence exist, load opcodeSet & Sentence...")

                with open(self.opcodeSetPath, "rb") as f:
                    opcodeSet = pickle.load(f)

                with open(self.sentencesPath, "rb") as f:
                    sentences = pickle.load(f)
            print("Number of opcodeSet: ", len(opcodeSet))
            
            print("Training word2vec model...")
            w2v = Word2Vec(sentences=sentences, vector_size=self.embeddingSize, window=5, min_count=0, workers=self.numWorkers)
            w2v.save(os.path.join(self.embeddingFolder, "opcode2vec.model"))
            print(f"Finish training word2vec model, save word2vec model to {self.embeddingFolder}")
        else:
            print("Word2vec model exist, load word2vec model...")
            w2v = Word2Vec.load(os.path.join(self.embeddingFolder, "opcode2vec.model"))

        return w2v

