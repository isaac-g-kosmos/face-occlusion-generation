import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('realOCC.csv')
# %%
# kepp only the path where the path starts with failed_aligned
df = df[df['path'].str.contains('failed_aligned')]

# %%
df['mask_path'] = df['path'].apply(lambda x: x.replace('jpg', 'png'))
df['mask_path'] = df['mask_path'].apply(
    lambda x: os.path.join(r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\fullmasks', x))
# %%
from PIL import Image

img = Image.open(df['mask_path'][0])
img = np.array(img)
plt.imshow(img)
plt.show()
# %%
df['mean_oclusion'] = 0
for x in range(len(df)):
    img = Image.open(df['mask_path'][x])
    bb = df[['bb_x1', 'bb_y1', 'bb_x2', 'bb_y2']].iloc[x].values
    img = img.crop(bb)
    img = np.array(img)
    img_mean = np.mean(img)
    df['mean_oclusion'].iloc[x] = img_mean / 255
# %%
df['mean_oclusion'].hist(bins=35)
plt.show()
# %%
sub_sample = df[df['mean_oclusion'] < 0.001]
saple = sub_sample.sample(1)
plt.imshow(Image.open(os.path.join(r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\FullrealOCC',
                                   saple['path'].values[0])))
plt.show()
img=Image.open(saple['mask_path'].values[0])
img=img.crop(saple[['bb_x1', 'bb_y1', 'bb_x2', 'bb_y2']].values[0])

plt.imshow(img)
plt.show()
# %%
hpad=pd.read_csv(r"C:\Users\isaac\PycharmProjects\tensorflow_filter\multitask_data\HPAD.csv")
#%%
hpad["mean_oclusion"]=0
hpad=hpad[['path','bb_x1', 'bb_y1', 'bb_x2', 'bb_y2','width','height','cara_cubierta','mean_oclusion']]
#%%
df1=pd.concat([df,hpad.sample(250)])
#drop mask path
df1=df1.drop(columns=['mask_path'])
df1.to_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\oocclusion_data\occtest.csv')
