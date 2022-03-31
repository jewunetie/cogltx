import pytorch_lightning as pl
from transformers import BertModel
import torch

from config import MAX_NUM_BLOCKS, DEFAULT_MODEL_NAME

class MemRecall(pl.LightningModule):
    def __init__(self, blocks, z_size=MAX_NUM_BLOCKS, upper_threshold=None, lower_threshold=None):
        self.blocks = blocks
        self.bert = BertModel.from_pretrained(DEFAULT_MODEL_NAME)
        self.linear = torch.nn.Linear(768, z_size)
        self.sigmoid = torch.sigmoid
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        
    def forward(self, ids):
        output = self.bert(ids)
        output = self.linear(output[0])
        output = self.sigmoid(output)
        
        
        
    def training_step(self, batch):
        pass
        
    