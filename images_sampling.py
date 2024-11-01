from PIL import Image
import numpy as np
oclussion=Image.open(r"C:\Users\isaac\PycharmProjects\high_quality_oclussion_aug\NatOcc_hand\mask\0.png")
oclussion=np.array(oclussion)
oclussion=oclussion.astype('uint8')
img=Image.open(r"C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebAMask-HQ-masks_corrected\0.png")
mask=np.array(img)/255
mask=mask.astype('uint8')[:,:,0]
print(np.max(oclussion))
print(np.max(mask))
#%%
import matplotlib.pyplot as plt
plt.imshow(mask)
plt.show()
plt.imshow(oclussion)
plt.show()
#%%
mask_mean=np.mean(mask)
oclussion_mean=np.mean(oclussion)
#%%
1-  oclussion_mean/mask_mean