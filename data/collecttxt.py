'''
这个代码是收集所有的txt文件到一起
'''
import os
rootpath = './LibriSpeech/train-clean-360/'
with open('./txt360.txt','a+') as f1:
    list1 = os.listdir(rootpath)
    for i in range(len(list1)):
        list2 = os.listdir(rootpath+'%s' % list1[i])
        for j in range(len(list2)):
            list3 = os.listdir(rootpath+'%s/%s' % (list1[i], list2[j]))
            for k in range(len(list3)):
                if 'txt' in list3[k]:
                    with open(rootpath+'%s/%s/%s' % (list1[i], list2[j], list3[k]),
                              encoding='cp936') as f2:
                        for line in f2:
                            f1.write(line)
                    f2.close()
f1.close()



