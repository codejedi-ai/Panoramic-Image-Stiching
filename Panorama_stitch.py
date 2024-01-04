import numpy as np
from PIL import Image
import matplotlib
import matplotlib.image as image
import matplotlib.pyplot as plt
from skimage.feature import (corner_harris, corner_peaks, plot_matches, BRIEF, match_descriptors)
from skimage.transform import warp, ProjectiveTransform
from skimage.color import rgb2gray
from skimage.measure import ransac
import os
folder = os.path.dirname(os.path.abspath(__file__))
# join the paths
matplot = os.path.join(folder, 'matplot')
if not os.path.exists('matplot'):
    os.makedirs('matplot')
print(folder)
print(matplot)
def blankImage(imL):
    imLalpha = imL.copy()
    imLalpha[:,:,0] = 0
    imLalpha[:,:,1] = 0
    imLalpha[:,:,2] = 0
    return imLalpha
def calculateAlpha(dtransP, dtransS):
    alpha =dtransP[:] / (dtransP[:] + dtransS[:])
    m = np.isnan(alpha) # mask of NaNs
    alpha[m] = 0
    return alpha
def applyAlpha(image,alpha):
    imagealpha = image.copy()
    imagealpha[:,:,0] = np.multiply(alpha,image[:,:,0])
    imagealpha[:,:,1] = np.multiply(alpha,image[:,:,1])
    imagealpha[:,:,2] = np.multiply(alpha,image[:,:,2])
    return imagealpha
def calculate_dtrans(image):
    length = image.shape[1]
    height = image.shape[0]

    a = np.fromfunction(lambda i, j: i,
                        (height, length),
                        dtype=int)
    b = np.fromfunction(lambda i, j: j,
                        (height, length),
                        dtype=int)
    c = height - 1 + np.fromfunction(lambda i, j: -i,
                                     (height, length),
                                     dtype=int)
    d = length - 1 + np.fromfunction(lambda i, j: -j,
                                     (height, length),
                                     dtype=int)

    # got no idea how to merge the 4 matricies please look at piazza @20
    mask = np.minimum(np.minimum(a, b), np.minimum(c, d)) + 1
    return mask / np.max(mask)
#  read "images/CMU_left.jpg" in pillow format
imL = image.imread("images/CMU_left.jpg")
imR = image.imread("images/CMU_right.jpg")
imLgray = rgb2gray(imL)
imRgray = rgb2gray(imR)

# NOTE: corner_peaks and many other feature extraction functions return point coordinates as (y,x), that is (rows,cols)
keypointsL = corner_peaks(corner_harris(imLgray), threshold_rel=0.0005, min_distance=5)
keypointsR = corner_peaks(corner_harris(imRgray), threshold_rel=0.0005, min_distance=5)

extractor = BRIEF()

extractor.extract(imLgray, keypointsL)
keypointsL = keypointsL[extractor.mask]
descriptorsL = extractor.descriptors

extractor.extract(imRgray, keypointsR)
keypointsR = keypointsR[extractor.mask]
descriptorsR = extractor.descriptors

matchesLR = match_descriptors(descriptorsL, descriptorsR, cross_check=True)
print ('the number of matches is {:2d}'.format(matchesLR.shape[0]))

fig = plt.figure(1,figsize = (12, 4))
axA = plt.subplot(111)
plt.gray()
plot_matches(axA, imL, imR, keypointsL, keypointsR, matchesLR) #, matches_color = 'r')
axA.axis('off')
# save the plt as an image
# if matplot folder does not exist, create it

# fig.savefig('matplot/P1.1-PLOT-matches-all.jpg', dpi=300)
# use os.path.join to join the paths
# give title to the plt
plt.title("matches between the left and right images")
fig.savefig(os.path.join(matplot, 'P1.1-PLOT-matches-all.jpg'), dpi=300)

enlargedL = blankImage(np.resize(imR,(imR.shape[0], imR.shape[1] * 2, 3)))
enlargedL[:,:imR.shape[1] ,0] = imL[:,:,0]
enlargedL[:,:imR.shape[1] ,1] = imL[:,:,1]
enlargedL[:,:imR.shape[1] ,2] = imL[:,:,2]

enlargedR = blankImage(np.resize(imR,(imR.shape[0] , imR.shape[1] * 2, 3)))
enlargedR[:,:imR.shape[1] ,0] = imR[:,:,0]
enlargedR[:,:imR.shape[1] ,1] = imR[:,:,1]
enlargedR[:,:imR.shape[1] ,2] = imR[:,:,2]

dst = np.roll( keypointsL[matchesLR[:,0]], 1, axis = 1)
src = np.roll( keypointsR[matchesLR[:,1]], 1, axis = 1)
#src = src + imL.shape[1]*np.column_stack((np.zeros(len(src)),np.ones(len(src))))
model_robust, inliers = ransac((src, dst),  ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=1000000)

fig = plt.figure(2,figsize = (12, 4))
axA = plt.subplot(111)
plt.gray()
plot_matches(axA, imL, imR, keypointsL, keypointsR, matchesLR[inliers]) #, matches_color = 'r')
axA.axis('off')
plt.title("matches between the left and right images (inliers)")
# fig.savefig('matplot/P2.1-PLOT-matches-inliers.jpg', dpi=300)
fig.savefig(os.path.join(matplot, 'P2.1-PLOT-matches-inliers.jpg'), dpi=300)

fig = plt.figure(3,figsize = (12, 14))

plt.subplot(311)

plt.imshow(enlargedL)
plt.title("reference frame with the left image")

plt.subplot(312)

transformedRight =  warp(enlargedR, model_robust.inverse)
transformedRight = (transformedRight * 255).astype(int)
plt.imshow(transformedRight)

plt.title("reference frame with the right image (reprojected)")


plt.subplot(313)
model_bad = ProjectiveTransform()
model_bad.estimate(src, dst)
plt.imshow( warp(enlargedR, model_bad.inverse))
plt.title("reference frame with the right image (reprojected badly)")

# fig.savefig('matplot/P1.2-PLOT-reprojected-matches-all.jpg', dpi=300)
fig.savefig(os.path.join(matplot, 'P1.2-PLOT-reprojected-matches-all.jpg'), dpi=300)


enlargedR_1 = blankImage(np.resize(imR,(imR.shape[0] , imR.shape[1]  * 2, 3)))
enlargedR_1[:,:imR.shape[1] ,0] = imR[:,:,0]
enlargedR_1[:,:imR.shape[1] ,1] = imR[:,:,1]
enlargedR_1[:,:imR.shape[1] ,2] = imR[:,:,2]
fig = plt.figure(4,figsize = (12, 3))
plt.subplot(121)
plt.title("dtrans1 in Ref. frame (LdtRef)")


imL_dtrans =np.zeros((imL.shape[0] , imL.shape[1]  * 2))
imL_dtrans[:,:imL.shape[1]] = calculate_dtrans(imL)

plt.imshow(imL_dtrans)


plt.subplot(122)
plt.title("dtrans2 in Ref. frame (RdtRef)")
imR_dtrans = imL_dtrans.copy()
imR_dtrans = warp(imR_dtrans, model_robust.inverse)
#plt.imshow(img_2_alpha)
plt.imshow( imR_dtrans)
# fig.savefig('matplot/APPDX-1.1-dtrans1-dtrans2.jpg', dpi=300)
fig.savefig(os.path.join(matplot, 'APPDX-1.1-dtrans1-dtrans2.jpg'), dpi=300)

alphaL = calculateAlpha(imL_dtrans,imR_dtrans)
alphaR = calculateAlpha(imR_dtrans,imL_dtrans)
fig = plt.figure(4,figsize = (12, 3))
plt.subplot(121)
plt.title("alpha1 in Ref. frame (LdtRef)")


plt.imshow(alphaL )


plt.subplot(122)
plt.title("alpha2 in Ref. frame (RdtRef)")

plt.imshow(alphaR)
# fig.savefig('matplot/APPDX-1.2-alpha1-alpha2.jpg', dpi=300)
fig.savefig(os.path.join(matplot, 'APPDX-1.2-alpha1-alpha2.jpg'), dpi=300)
fig = plt.figure(5, figsize=(12, 14))
plt.subplot(311)

img_1 = applyAlpha(enlargedL, alphaL)

plt.imshow(img_1)

plt.title("alpha based on DT (for RransacRef)")

plt.subplot(312)
# plt.imshow(...)

img_2R = applyAlpha(transformedRight, alphaR)

plt.imshow(img_2R)
plt.title("alpha based on DT (for RransacRef)")

plt.subplot(313)
Panorama = img_2R + img_1
plt.imshow(Panorama)
plt.title("Panorama")

# plt.show()
# fig.savefig('matplot/P2.2-PLOT-reprojected-matches-inliers.jpg', dpi=300)
fig.savefig(os.path.join(matplot, 'P2.2-PLOT-reprojected-matches-inliers.jpg'), dpi=300)
# save the Panorama as an image
# if Panorama folder does not exist, create it
if not os.path.exists('Panorama'):
    os.makedirs('Panorama')
# use pillow to save the image
Panorama = Image.fromarray(Panorama.astype(np.uint8))
Panorama.save('Panorama/Panorama.jpg')