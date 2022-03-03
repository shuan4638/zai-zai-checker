import pprint
import numpy as np
from transformers import BertModel, BertTokenizer, pipeline, logging
logging.set_verbosity_error()

unmasker = pipeline('fill-mask', model="bert-base-chinese")

txt = '閃尿逃兵整天對中國嗆，真的戰爭來在推年輕人去死。'
txt_mask = txt.replace('在', '[MASK]')
txt_unmask = unmasker(txt_mask)
print("你是不是想說:")
for i in range(8):
    sequence = txt_unmask[i]['sequence'].replace(' ', '')
    score = round(txt_unmask[i]['score'], 3)
    print(f'{i+1}. {sequence} score: {score}')
