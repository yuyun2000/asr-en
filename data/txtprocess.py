'''
这个代码是处理生成的TXT 主要是去掉前面的文件名称 以及所有字体变小写
'''
with open('./txt360.txt') as f1:
    for line in f1:
        neme = line.split()
        with open('./train360.txt','a+') as f2:
            f2.write(neme[0]+'\t')
            for i in range(len(neme)-1):
                f2.write(neme[i+1].lower()+" ")
            f2.write('\n')
f2.close()
f1.close()
