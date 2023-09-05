import torch
from torchtext.vocab import GloVe
import numpy as np
import os
import time
import lstm_utils as utils
from itertools import product
directory = os.path.dirname(__file__)
#tensor1 = np.random.randint(0, 10, size=(2,3,3))
#tensor2 = np.random.randint(0, 10, size=(2,3,4))
#res = torch.nn.utils.rnn.pad_sequence([torch.tensor(tensor1),torch.tensor(tensor2)],batch_first=True)

file = os.path.join(directory,"glove.6B/glove.6B.50d.txt")
embeddings = GloVe(name="6B", cache = os.path.join(directory,"glove.6B"),dim = 50)

stack = []
frase = [["hello","how","are"],["fine","thanks","you","asshole"]]
for fr in frase:
    stack.append(embeddings.get_vecs_by_tokens(fr))
    
stack =  torch.nn.utils.rnn.pad_sequence(stack,batch_first=True,padding_value=-100)

print(stack,stack.size())
hyp_comb = list(product(utils.LEARNING_RATE, utils.DROPOUT_LAYER, utils.LIN_WD))
print(type(hyp_comb))
