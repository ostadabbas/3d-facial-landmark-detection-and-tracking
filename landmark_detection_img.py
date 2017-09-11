#!/usr/bin/python
import numpy as np
import cv2
from skimage import io
import helpers
from matplotlib import pyplot as plt


######################################################
## Landmark detection with one image.               ##
## The result is saved as a PDF file.               ##
######################################################

######################## read image ################################
# For one image
im = io.imread("./dataSample/Yu.jpg")
(bb, coords) = helpers.get_landmarks(im)

#np.savetxt('landmark_result_Yu.txt', coords)


########################## normalize ###############################
desiredEyePixels = 150
(im_norm, landmarks_norm) = helpers.normalize(im, coords, desiredEyePixels)


################### output landmark coordinates ####################
for i in [0,16,36,43]:#np.arange(68):
    print "The", i,"th landmark:", coords[i]
  

######################## visualization #############################
highlights = [22,33,48]
outImg = helpers.visualize_facial_landmarks(im, bb, coords, 1,highlights)
outImg_noBackground = helpers.visualize_facial_landmarks(im, bb, coords, 0,highlights)
outImg_norm = helpers.visualize_facial_landmarks(im_norm, bb, landmarks_norm, 1,highlights)

plt.subplot(121), plt.imshow(outImg)
plt.subplot(122), plt.imshow(outImg_noBackground)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

fig = plt.figure()
outImg1 = helpers.visualize_facial_landmarks(im, bb, coords, 1,[])
plt.imshow(outImg1)
plt.imshow(outImg_norm)

plt.show()

fig.savefig('Yu1.pdf')
