英文asr
-

- 整体思路和之前的中文asr相同
- 分词采用bpe分词，huggingface的token有库可以调用，只需要自己将语料内容集中在一个文本里即可，详细见data文件夹下的操作，最终分成500个subword
- 模型输入是1600*80*1，大概是十几秒的语音，运行也比较慢，但是应该可以实时
- 之前训练的模型因为是直接在原始数据集上训练也没有加增强，所以出来的结果只在原始数据集上效果还行，后面加增强训练太耽误事了还没训出来
- 增强的时候虽然要预先计算好频谱图，但是保存的时候不要除以255，以uint8保存成图，这样保存和加载都快，或者保存成一个大的npy，不过我不太喜欢这样
