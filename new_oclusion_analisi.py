from PIL import Image
import numpy as np
import os
#%%
hands_path=r"C:\Users\isaac\PycharmProjects\high_quality_oclussion_aug\NatOcc_hand\mask"
obj_path=r'C:\Users\isaac\PycharmProjects\high_quality_oclussion_aug\NatOcc_objects\mask'
text_path=r'C:\Users\isaac\PycharmProjects\high_quality_oclussion_aug\RandOcc\mask'
hands_=os.listdir(hands_path)
obj_=os.listdir(obj_path)
text_=os.listdir(text_path)
masks=[os.path.join(hands_path,x) for x in hands_]
obj_mask=[os.path.join(obj_path,x) for x in obj_]
text_mask=[os.path.join(text_path,x) for x in text_]
masks.extend(obj_mask)
masks.extend(text_mask)
#%
base_paths=[os.path.basename(x) for x in masks]
base_paths=[os.path.join(r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebAMask-HQ-masks_corrected',
                         x) for x in base_paths]
#%%
oclussion_percent=[]
for x in range(len(masks)):
    print(x)
    oclussion = Image.open(masks[x])
    oclussion = np.array(oclussion)
    oclussion = oclussion.astype('uint8')
    mask = Image.open(base_paths[x])
    mask = np.array(mask) / 255
    mask = mask.astype('uint8')[:, :, 0]
    mask_mean = np.mean(mask)
    oclussion_mean = np.mean(oclussion)
    # %%
    oclussion_percent.append(1 - oclussion_mean / mask_mean)
#%%
import pandas as pd
df=pd.DataFrame({'path':masks,'oclussion':oclussion_percent})
#%%
def partition_formula(path):
    if 'NatOcc_hand' in path:
        return 'NatOcc_hand'
    if 'NatOcc_objects' in path:
        return 'NatOcc_objects'
    if 'RandOcc' in path:
        return 'RandOcc'
df['type']=df['path'].apply(lambda x:partition_formula(x))
df['img_path']=df['path'].apply(lambda x:x.replace('mask','img'))
df['img_path']=df['path'].apply(lambda x:x.replace('mask','img'))
df['img_path']=df['img_path'].apply(lambda x:x.replace('png','jpg'))
#%%
df.to_csv('oclusion_augmentations.csv')
#%%
import matplotlib.pyplot as plt


df['oclussion'].hist(bins=10, label='10 bins')
df['oclussion'].hist(bins=20, label='20 bins')
df['oclussion'].hist(bins=30, label='30 bins')
df['oclussion'].hist(bins=50, label='50 bins')

plt.legend()
plt.show()
#%%
hands_=df[df['type']=='NatOcc_hand']
obj=df[df['type']=='NatOcc_objects']
textutes=df[df['type']=='RandOcc']

obj['oclussion'].hist(bins=20, label='Obj 20 bins')
hands_['oclussion'].hist(bins=20, label='Hand 20 bins')
textutes['oclussion'].hist(bins=10, label='Texture 20 bins')

plt.legend()
plt.show()
#%%
low_percent_oclussionn=df[(df['oclussion']<.15)&(.1< df['oclussion'])]
# low_percent_oclussionn=low_percent_oclussionn[low_percent_oclussionn['type']=='RandOcc']
print(low_percent_oclussionn['type'].value_counts())
sample=low_percent_oclussionn.sample(5)
imgs=sample['img_path'].tolist()
masks=sample['path'].tolist()
oclussi_per=sample['oclussion'].tolist()
fig, ax = plt.subplots(2, 5)


for j in range(5):
    try:
        path = imgs[j]
        img = Image.open(path)

        path = masks[j]
        mask= Image.open(path)
        ax[0, j].imshow(img)
        ax[1, j].text(0, 0, np.round(oclussi_per[j],2),style='italic')
        ax[0, j].axis('off')
        ax[1, j].imshow(mask)
        ax[1, j].axis('off')
    except Exception as e:
        print(e)
plt.show()
#%%

df['type'].value_counts()

#%%%
# obj.hist()
# plt.show()
obj_small=obj[obj['oclussion']<.10]
hands_small=hands_[hands_['oclussion']<.1]
obj_normal_dist=obj[obj['oclussion']>.1].sample(2000)
hands_normal_dist=hands_[hands_['oclussion']>.1].sample(2000)
df_final=pd.concat([textutes,obj_small,hands_small,obj_normal_dist,hands_normal_dist])
df_final['oclussion'].hist(bins=20)
#%%
df_final.to_csv('final_oclusion_augs.csv',index=False)
#%%
df['oclussion'].hist(bins=20,label='Oclussion 20 bins')
plt.legend()
plt.show()