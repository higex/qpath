import sys
sys.path.append('/Users/chief/higex/qpath/')

from util.storage import ModelPersistence

with ModelPersistence('/Users/chief/higex/qpath/models/rgb_tissue_models.pck', 'c', format='pickle') as d:
    rgb_tissue_models = d['models']
    
import matplotlib.pylab as plt
plt.gray()

from math import sqrt
import numpy as np

import mahotas

from skimage.filters import *
from skimage.filters.rank import *
from skimage.io import *
from skimage.morphology import *
from skimage.feature import blob_doh, blob_dog
from skimage.measure import *

from sklearn.ensemble import *
from sklearn.tree import *

# QPATH
#from descriptors.txtgrey import *
#from descriptors.txtbin import *
#from descriptors.extract import *

from stain.he import *
from util.storage import *
from util.intensity import *
from segm.tissue import *

#im_name = '/Users/chief/higex/tst/tst-nuclei-01.ppm'
nn = '_n02'
mg = '_40x'
nm = '0002-G5001'

im_name = '/Users/chief/data/CRC/' + nm + mg + nn + '.ppm'

im = imread(im_name)
im_h, im_e, _ = rgb2he2(im)


# superpixels
#sp = slic(im, n_segments=1000, compactness=50, sigma=1.5, multichannel=True, convert2lab=True)
#
#n_sp = sp.max() + 1
#img_res = np.ndarray(im.shape, dtype=im.dtype)
#
#for i in np.arange(n_sp):
#    img_res[sp == i, 0] = int(np.mean(im[sp == i, 0]))
#    img_res[sp == i, 1] = int(np.mean(im[sp == i, 1]))
#    img_res[sp == i, 2] = int(np.mean(im[sp == i, 2]))
#
#plt.imshow(img_res)


map = tissue_components(im, rgb_tissue_models)

im_n = im_h * (map == 1)
im_h2 = threshold_adaptive(im_h, 211)

th = threshold_otsu(im_n)
im_h2 = im_n > th
im_h2 = mahotas.open(im_h2)
im_h2 = remove_small_objects(im_h2, in_place=True)
im_h2 = mahotas.close_holes(im_h2, Bc=np.ones((5,5)) )


im_h1 = gaussian_filter(im_h, 1.5)

im_h1 = im_h.copy()
th = threshold_otsu(im_h1)
im_h2 = mahotas.close_holes(im_h1 >= th, Bc=np.ones((5,5)) )
im_h2 = remove_small_objects(im_h2, in_place=True)
im_n2 = im_h * im_h2


dst  = mahotas.stretch(mahotas.distance(im_h2))
Bc   = np.ones((9,9))
lmax = mahotas.regmax(dst, Bc=Bc)
spots, _ = mahotas.label(lmax, Bc=Bc)
regions = mahotas.cwatershed(lmax.max() - lmax, spots) * im_h2

regions = relabel(regions)
    
reg_props = regionprops(regions, im_h1)

regs = regions.copy()
for r in reg_props:
    if r.area < 100:
        regs[regions == r.label] = 0
        continue
    if r.solidity < 0.1:
        regs[regions == r.label] = 0
        continue
    if r.minor_axis_length / r.major_axis_length < 0.1:
        regs[regions == r.label] = 0
        continue
regs = relabel(regs)
reg_props = regionprops(regs, im_h1)
        
        
#blobs_doh = blob_doh(im_n2, max_sigma=30, threshold=.01)
#blobs = blobs_doh

blobs_dog = blob_dog(im_n2, min_sigma=2.5, max_sigma=7.5, threshold=.01)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs = blobs_dog
lb = np.zeros((blobs.shape[0],1))
for k in np.arange(blobs.shape[0]):
    y, x = blobs[k,0:2]
    lb[k, 0] = regs[y, x]
blobs = np.concatenate((blobs, lb), axis=1)

final_blobs = []
for r in reg_props:
    yc, xc = r.centroid
    idx = np.where(blobs[:,3] == r.label)[0]
    if len(idx) == 0:
        continue
    d = np.zeros(len(idx))
    for i in range(len(idx)):
        y, x = blobs[idx[i],0:2]
        d[i] = (x-xc)**2 + (y-yc)**2
    i = np.argmin(d)
    final_blobs.append(blobs[idx[i],])
    
fig, ax = plt.subplots(1, 1)
ax.imshow(im, interpolation='nearest')
for blob in final_blobs:
    y, x, r, _  = blob
    if regs[y,x] == 0:
        continue
    regs[regs == regs[y,x]] = 0
    r = max(1, r)
    c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=True)
    ax.add_patch(c)

plt.show()



def relabel(regs):
    res = np.zeros(regs.shape, dtype=np.int64)
    r = np.unique(regs.ravel())
    for i in range(1, r.size):
        res[regs == r[i]] = i
        
    return res
    

# RAG
from skimage import segmentation
from skimage.future import graph
from matplotlib import colors


img = im * np.im_h
img = im * np.repeat(np.expand_dims(im_h2, axis=2), 3, axis=2)

labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)

out = graph.draw_rag(labels, g, img)
plt.figure()
plt.title("RAG with all edges shown in green.")
plt.imshow(out)

# The color palette used was taken from
# http://www.colorcombos.com/color-schemes/2/ColorCombo2.html
cmap = colors.ListedColormap(['#6599FF', '#ff9900'])
out = graph.draw_rag(labels, g, img, node_color="#ffde00", colormap=cmap,
                     thresh=30, desaturate=True)
plt.figure()
plt.title("RAG with edge weights less than 30, color "
          "mapped between blue and orange.")
plt.imshow(out)

plt.figure()
plt.title("All edges drawn with cubehelix colormap")
cmap = plt.get_cmap('cubehelix')
out = graph.draw_rag(labels, g, img, colormap=cmap,
                     desaturate=True)

plt.imshow(out)
plt.show()
