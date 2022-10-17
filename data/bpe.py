'''
这个代码是BPE分词器
'''
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())

from tokenizers.pre_tokenizers import ByteLevel

tokenizer.pre_tokenizer = ByteLevel()

from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=[ "[PAD]"],vocab_size=500)
tokenizer.train(files=["all.txt"], trainer=trainer)
tokenizer.save("./500hbpe-500.json")

output = tokenizer.encode("hello my friend how are you today?")
print(output.tokens)
print(output.ids)

