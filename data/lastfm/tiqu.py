import os
path='./remapkg1.csv'
# cnt=0
# num=0
# with open(path,'r') as f:
#     for line in f:
#         line=line.strip().split(',')
#         cnt+=1
#         user,_, item = int(line[0]), int(line[1]), int(line[2])
#         num=max(num,user)
#         num=max(num,item)
# print('cnt== ',cnt)
# print('num == ',num)
dr={}
di={}
with open(path,'r') as f:
    for line in f:
        line=line.strip().split(',')
        user,_,item=line[0],line[1],line[2]
        with open('./train2id.txt','a') as f1:
            f1.write(user+'\t'+item+'\t'+_+'\n')
        user,_, item = int(line[0]), int(line[1]), int(line[2])
        if(di.get(user) is None):
            di[user]=user
            with open('./entity2id.txt','a') as f1:
                f1.write(str(user)+'\t'+str(user) +'\n')
        if(dr.get(_) is None):
            dr[_]=_
            with open('./relation2id.txt','a') as f1:
                f1.write(str(_)+'\t'+str(_) +'\n')
        if(di.get(item) is None):
            di[item]=item
            with open('./entity2id.txt','a') as f1:
                f1.write(str(item) +'\t'+str(item) +'\n')