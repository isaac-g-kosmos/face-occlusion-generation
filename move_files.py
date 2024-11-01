import os
txt_path=r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebAMask-HQ-WO-train.txt'
with open(txt_path,'r') as f:
    lines=f.readlines()

import shutil
input_path=r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebAMask-HQ-masks_corrected'
output_path=r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebAMask-HQ-WO-Train_mask'
#%%
for x in lines:
    path=os.path.join(input_path,x.strip().replace('jpg','png'))
    output=os.path.join(output_path,x.strip().replace('jpg','png'))
    shutil.copy(path,output)