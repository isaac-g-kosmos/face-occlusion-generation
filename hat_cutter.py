import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebAMask-HQ\CelebAMask-HQ-attribute-anno.txt',
    sep=' ', index_col=False)
# %%
hats = df[df['Wearing_Hat'] == 1]
# %%
path = r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebAMask-HQ\CelebAMask-HQ-mask-anno'
import os

all_paths = []
for x in range(15):
    new_path = os.path.join(path, str(x))
    for y in os.listdir(new_path):
        if y.endswith('hat.png'):
            all_paths.append(os.path.join(new_path, y))
# %%
from PIL import Image

pictures_paths = r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\dataset\CelebA-HQ-img'
hats['path'] = hats['path'].apply(lambda x: os.path.join(pictures_paths, x))
samples = hats.sample(10)

paths = samples['path'].to_list()

fig, ax = plt.subplots(2, 5)

for i in range(2):
    for j in range(5):
        path = paths[j * 2 + i]
        img = Image.open(path)
        # ax[i, j].text(0,0, np.round(samples['prob'].values[j*2+i],2), style='italic')

        ax[i, j].imshow(img)
        ax[i, j].axis('off')

plt.show()
# %%
all_paths_base = [str(int(os.path.basename(x).replace('_hat.png', ''))) + '.jpg' for x in all_paths]
all_path_dir = [os.path.dirname(x) for x in all_paths]
# %%
output_path=r'C:\Users\isaac\PycharmProjects\face-occlusion-generation\hat_overlays'
for x in range(len(hats)):
    try:
        sample_path=hats['path'].iloc[x]
        base_path=os.path.basename(sample_path)
        index=all_paths_base.index(base_path)
        overlay_path=all_paths[index]

        img=Image.open(sample_path)
        alpha_image = Image.new("RGBA", img.size, (255, 255, 255, 0))
        # plt.imshow(img)
        # plt.show()
        overlay=Image.open(overlay_path)
        # plt.imshow(overlay)
        # plt.show()
        mask = overlay.convert("L")
        mask=mask.resize(img.size)
        # Apply the mask to the original image
        result = Image.alpha_composite(img.convert("RGBA"), alpha_image)

        mark_array=np.array(mask)
        indeces=np.where(mark_array==255)
        x_max=int(np.max(indeces[0]))
        y_max=int(np.max(indeces[1]))
        x_min=int(np.min(indeces[0]))
        y_min=int(np.min(indeces[1]))
        #
        # print(result.size)
        # print(mask.size)
        result.putalpha(mask)
        result=result.crop([y_min,x_min,y_max,x_max])
        # plt.imshow(result)
        # plt.show()
        result.save(os.path.join(output_path,f'{x}.png'))
    except:
        pass