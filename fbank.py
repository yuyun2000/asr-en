'''
这个代码是把flac文件预处理为npy保存下来
fbank特征超过1600的，插值返回1600不影响原来的内容，语速会变快
fbank长度小于1600，在后面补0
'''
import numpy as np
import cv2
from prepro import compute_log_mel_fbank
import os


import os
import cv2
rootpath = './data/all/'
prodata = './data/allpic/'
listold = os.listdir(prodata) #确定已有的文件，不重复生成
for i in range(len(listold)):
    listold[i] = listold[i][:-4]
print('load over')
# print(listold)

list1 = os.listdir(rootpath)
for i in range(len(list1)):
                if 'txt' not in list1[i] and list1[i][:-4] not in listold:
                    file = rootpath+'%s' % (list1[i])
                    out = compute_log_mel_fbank(file)
                    if out.shape[0] >=1600:
                        out = cv2.resize(out, (80, 1600), interpolation=cv2.INTER_AREA)
                    else:
                        out = np.pad(out,((0,1600-out.shape[0]),(0,0)))
                    cv2.imwrite('./data/allpic/%s.jpg'%list1[i][:-5],out)
                    # np.save('D:/libri360-noise-pro/%s'%list1[i][:-4],out)

