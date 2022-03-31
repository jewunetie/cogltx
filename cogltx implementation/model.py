import pytorch_lightning as pl
from transformers import BertModel
import torch

from config import MAX_NUM_BLOCKS, DEFAULT_MODEL_NAME, LABELS

class SDGTagger(pl.LightningModule):
    def __init__(self, blocks, num_classes=len(LABELS), z_size=MAX_NUM_BLOCKS, upper_threshold=None, lower_threshold=None):
        self.blocks = blocks
        self.judge_bert = BertModel.from_pretrained(DEFAULT_MODEL_NAME)
        self.judge_linear1 = torch.nn.Linear(self.judge_bert.config.hidden_size, z_size)
        self.judge_linear2 = torch.nn.Linear(z_size, num_classes)
        self.sigmoid = torch.sigmoid
        
        self.reasoner_bert = BertModel.from_pretrained(DEFAULT_MODEL_NAME)
        self.reasoner_classifier = torch.nn.Linear(self.reasoner_bert.config.hidden_size, num_classes)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        
    def calculate_relevance(self, z):
        for index in z:
            z_copy = z.copy()
            z.remove(index)
            
            
            
        
    def reasoner_forward(self, input_block_indexes):
        output = self.reasoner_bert(inputs_ids)
        output = self.reasoner_classifier(output[0])
        output = self.sigmoid(output)
        return output
        
    def judge_forward(self, input_block_indexes):
        output = self.judge_bert(inputs_ids)
        output = self.judge_linear(output[0])
        output = self.sigmoid(output)
        return output
        
    def training_step(self, batch):
        pass
        
    