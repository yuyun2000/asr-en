'''
修改ir增强后的npy文件和原来的区别开，并且修改txt文件里的文件名
'''
import os
# npylist = os.listdir('D:/libri360-noise-pro')
# for i in range(len(npylist)):
#     os.rename('D:/libri360-noise-pro/%s'%npylist[i],'D:/libri360-noise-pro/%s-noise.npy'%npylist[i][:-4])


with open('ir360.txt','r') as f:
    with open('noise360.txt','a+') as f2:
        for line in f:
            name,text = line.split('\t')
            name = name[:-3]+'-noise'
            f2.write(name+'\t'+text)
f.close()
f2.close()

