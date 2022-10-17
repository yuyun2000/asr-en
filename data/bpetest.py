'''
这个代码是将英文的语料变成训练用的id
'''
import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("500hbpe-500.json")

output = tokenizer.encode("china")
print(output.tokens)
print(output.ids)


# l = []
# with open('./train.txt') as f:
#     for line in f:
#         name,text = line.split('\t')
#         encoded = tokenizer.encode(text)
#         l.append(len(encoded.ids))
#
# print(max(l))
# print(np.mean(l))
# print(np.median(l))
# f.close()