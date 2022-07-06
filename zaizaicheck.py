import pprint
import numpy as np
from transformers import BertModel, BertTokenizer, pipeline, logging
logging.set_verbosity_error()

unmasker = pipeline('fill-mask', model="bert-base-chinese")

txt = '我現在不想聽，請你不要在說了。'
txt_mask = txt.replace('在', '[MASK]')
txt_unmask = unmasker(txt_mask)
print("你是不是想說:")
for i in range(8):
    sequence = txt_unmask[i]['sequence'].replace(' ', '')
    score = round(txt_unmask[i]['score'], 3)
    print(f'{i+1}. {sequence} score: {score}')
