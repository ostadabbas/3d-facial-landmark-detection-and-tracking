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


################### output landmark coordinates ####################
for i in [0,16,36,43]:#np.arange(68):
    print("The " +  str(i) + "th landmark:")
    print(coords[i])
  

######################## visualization #############################
highlights = [16,36,43]
outImg = helpers.visualize_facial_landmarks(im, bb, coords, 1,highlights)
outImg_noBackground = helpers.visualize_facial_landmarks(im, bb, coords, 0,highlights)

plt.subplot(121), plt.imshow(outImg)
plt.subplot(122), plt.imshow(outImg_noBackground)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

fig = plt.figure()
outImg1 = helpers.visualize_facial_landmarks(im, bb, coords, 1,[])
plt.imshow(outImg1)

plt.show()

fig.savefig('Yu1.pdf')
