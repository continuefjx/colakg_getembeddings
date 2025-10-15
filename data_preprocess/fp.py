import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from collections import defaultdict
path='../data/lastfm/remapkg1.csv'
user_item_path='../data/lastfm/train.txt'
item_embedding_path='../data/lastfm/embeddings.pt'
item_embedding=torch.load(item_embedding_path)
item_embed,attri_embed=torch.split(item_embedding,[2813,item_embedding.shape[0]-2813],dim=0)
user_item=defaultdict(set)
cnt=1
with open(user_item_path,'r') as f:
    for line in f:
        # print(cnt)
        # cnt+=1
        line=line.strip().split(' ')
        user,item=int(line[0]),list(map(int,line[1:]))
        # print(user)
        # print(item)
        user_item[user].update(item)
item_dict=defaultdict(set)
with open(path,'r') as f:
    for line in f:
        line=line.strip().split(',')
        user,_, item = int(line[0]), int(line[1]), int(line[2])
        item_dict[user].add(item)
print('read over\n')
import random
row=[]
lie=[]
val=[]
# rate=0.5
for key,values in user_item.items():
    values=list(values)
    # value=random.sample(values,int(2*rate*len(values)))
    for i in values:
        j1=list(item_dict[i])
        # lt=random.sample(j1,int(rate*len(j1)))
        for j in j1:
            row.append(key)
            lie.append(j-2813)
            val.append(1.0)
print('coomat\n')
print(max(row))
print(max(lie))
mat=coo_matrix((val,(row,lie)),dtype=np.float64)
dense=mat.toarray()
print('dense\n')
sum_mat=np.sum(dense,axis=1,keepdims=True)
print(sum_mat.shape,'summat\n')
print(dense.shape,'dense\n')
print(attri_embed.shape,'attri_embed\n')
print(item_embed.shape,'item_embed\n')
user_emb=(mat @attri_embed.cpu().detach().numpy())/sum_mat
user_emb=torch.from_numpy(user_emb).type(torch.float32)
print(user_emb.dtype)
print('baocun\n')
torch.save(user_emb,'../data/lastfm/user_emb.pt')
print('over\n')


