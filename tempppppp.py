import time
from transformers import *
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
# model= AlbertModel.from_pretrained('albert-base-v1')

start_time = time.time()
model(torch.randint(0, 30000, (4, 289)))
end_time = time.time()
print("Total execution time: {} seconds".format(end_time - start_time))
