import os
import numpy
file = os.listdir('D:/libri360-noise-pro')
# num = 0
# with open('./ir360.txt') as f:
#     for line in f:
#         name,_ = line.split('\t')
#         if name+'.npy' in file:
#             num += 1
# f.close()
# print(num)
for i in file:
    x = numpy.load('D:/libri360-noise-pro/%s'%i)
