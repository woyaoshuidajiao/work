import tifffile as tiff
import numpy as np
from skimage import io
from PIL import Image

img = tiff.imread('./R-C(1).tif')
print(img.shape)

# img = img.transpose([1, 2, 0])


col = 0
n = 0
while(col + 100 < 494):
    tiff.imsave('./image_%d.tif'%n, img[ :, n*100:(n+1)*100])
    col = col + 100
    n = n+1

tiff.imsave('./image_%d.tif'%n, img[ :, -100:])

# tiff.imsave('./image_1.tif', img_1)
# print(type(img))
# print(img.shape)
