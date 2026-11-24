"""Invent squares, not actual cars or a detection-quality benchmark."""
from pathlib import Path

import numpy as np
from PIL import Image

path=Path('synthetic-demo')
if path.exists():raise ValueError('synthetic-demo exists; no overwrite.')
path.mkdir();rows=['image,xmin,ymin,xmax,ymax,group']
for i in range(3):
    image=np.zeros((32,32,3),dtype=np.uint8);image[8:24,8:24]=255
    Image.fromarray(image).save(path/f'image{i}.png')
    rows.append(f'image{i}.png,8,8,24,24,invented-video-{i}')
(path/'boxes.csv').write_text('\n'.join(rows)+'\n',encoding='utf-8')
print(path/'boxes.csv')
