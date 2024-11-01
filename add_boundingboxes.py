import pandas as pd
import os
df=pd.read_csv('final_oclusion_augs.csv')
df['path']=df['img_path'].apply(lambda x:os.path.basename(x))
bb_df=pd.read_csv('CELEB_mask.csv')
#%%
df_final=pd.merge(df,bb_df,how='left',on='path')
#%%
df_final.dropna(inplace=True)
#%%
df_final['path']=df_final['img_path'].apply(lambda x:x.replace('C:\\Users\\isaac\\PycharmProjects\\high_quality_oclussion_aug\\',''))
df_final['path']=df_final['path'].apply(lambda x:x.replace('\\','/'))

#%%
df_final=df_final[['path','bb_x1','bb_y1','bb_x2','bb_y2','width','height','cara_cubierta','oclussion']]
df_final.to_csv(r'C:\Users\isaac\PycharmProjects\tensorflow_filter\oocclusion_data\HQFO_augmentaions.csv',index=False)




