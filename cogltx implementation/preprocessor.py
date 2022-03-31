import stanza
import re
from transformers import AutoTokenizer
import nltk

from config import DEFAULT_MODEL_NAME, MAX_BLOCK_SIZE


nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)
DEFAULT_MODEL_NAME = 'roberta-base'

class Block:
    def __init__(self, pos, ids=None, sep_token=None):
        if ids is None:
            ids = []
        self.ids = ids
        self.pos = pos
        self.sep_token = sep_token
        
        self.relevance = 0
        self.estimation = 0
    
    def add_sentence_ids(self, ids):
        self.ids.extend(ids)
        
    def init_relevance(self, label_desc):
        # get the cosine similarity between the label description and the ids
        sentence = str(self.ids)
        
    def __lt__(self, rhs):
        return self.blk_type < rhs.blk_type or (self.blk_type == rhs.blk_type and self.pos < rhs.pos)
    
    def __ne__(self, rhs):
        return self.pos != rhs.pos or self.blk_type != rhs.blk_type
    
    def __len__(self):
        return len(self.ids)
    
    # def __str__(self):
    #     return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))

class BlockArray():
    def __init__(self, blocks = None) -> None:
        if blocks is None:
            blocks = []
        self.blocks = blocks
        self.next_position = 0
        
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        self.CLS_TOKEN = self.tokenizer.cls_token_id
        self.SEP_TOKEN = self.tokenizer.sep_token_id
        self.PAD_TOKEN = self.tokenizer.pad_token_id
        
        self.block_size = MAX_BLOCK_SIZE
        
    def __getitem__(self, index):
        return self.blocks[index]
        
    def get_blocks(self, indexes):
        return [self[index] for index in indexes]
    
    def create_index_list(self):
        return [block.pos for block in self.blocks]
        
    # split text into sentences
    def split_text_into_sentences(self, text):
        # clean the text by removing unnecessary characters
        text = re.sub(r'\s+', ' ', text)
        
        # split text into sentences
        sentences = nltk.tokenize.sent_tokenize(text)
        return sentences
    
    def add_padding_tokens(self, block):
        pad_len = self.block_size - len(block)
        if pad_len > 0:
            padding = [self.PAD_TOKEN] * pad_len
            block.add_sentence_ids(padding)

    # group a list of sentences into blocks of size BLOCK_SIZE
    def group_sentences_into_blocks(self, sentences):
        current_block = Block(self.next_position, sep_token=self.SEP_TOKEN)
        for sentence in sentences:
            tokenized_sentence = self.tokenzier.encode(sentence)
            if len(current_block) + len(tokenized_sentence) <= self.block_size:
                current_block.add_sentence_ids(tokenized_sentence)
            else:
                if len(current_block) > 0:
                    self.blocks.append(current_block)
                    self.next_position += 1
                    current_block = Block(self.next_position, sep_token=self.SEP_TOKEN)
                
                if len(tokenized_sentence) > self.block_size:
                    tokenized_sentence = tokenized_sentence[:self.block_size]
                    print(f"Truncation at position {self.next_position} with sentence '{sentence}'")
                current_block = Block(tokenized_sentence, self.next_position)
        self.blocks.append(current_block)