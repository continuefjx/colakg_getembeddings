import torch
user_emb_path='./user_emb.pt'
item_emb_path='./embeddings.pt'
llm_user_emb_path='./lastfm_embeddings_simcse_kg_user.pt'
llm_item_emb_path='./lastfm_embeddings_simcse_kg.pt'
finuser_emb_path='./user_finemb.pt'
finitem_emb_path='./item_finemb.pt'

# finuser=torch.load(user_emb_path)
# print(finuser.shape)
#
#
# finuser=torch.cat([finuser,finuser],dim=1)
# #finuser=torch.cat([finuser,finuser],dim=1)
#
# print(finuser.dtype)
# llm_user_emb=torch.load(llm_user_emb_path)
# print(llm_user_emb.shape)
# print(llm_user_emb.dtype)
# llm_user_emb=llm_user_emb.to(finuser.device)+finuser
# torch.save(llm_user_emb,finuser_emb_path)
# print('user emb saved')

item_emb=torch.load(item_emb_path)
item_emb,_=torch.split(item_emb,[2813,item_emb.shape[0]-2813])
#item_emb=torch.cat([item_emb,item_emb],dim=1)
print(item_emb.shape)
item_emb=torch.cat([item_emb,item_emb],dim=1)
print(item_emb.shape)
llm_item_emb=torch.load(llm_item_emb_path)
llm_item_emb=llm_item_emb.to(item_emb.device) +item_emb
torch.save(llm_item_emb,finitem_emb_path)
print('over')