'''
这个文件是转移libri的录音到同一个文件夹内
'''
import os
import shutil

rootpath = './LibriSpeech/train-clean-360/'

list1 = os.listdir(rootpath)
for i in range(len(list1)):
        list2 = os.listdir(rootpath+'%s' % list1[i])
        for j in range(len(list2)):
            list3 = os.listdir(rootpath+'%s/%s' % (list1[i], list2[j]))
            for k in range(len(list3)):
                if 'txt' not in list3[k]:
                    shutil.move(rootpath+'%s/%s/%s'%(list1[i],list2[j],list3[k]),'./all/%s'%list3[k])

