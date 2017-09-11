#!/usr/bin/python
import numpy as np
import cv2
from skimage import io
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import math

import helpers
import head_move_box
def dist(ar):
    return np.sqrt((ar[0] ** 2) + (ar[1] ** 2))

#parameter setting
movPt = [22,43,48] #22 eyebrow, 33 nose, 43 upper eyelip, 47 lower eyelip, 48 lips
nameOfMovPT = ["Eyebrow Inner (mm)", "Eye Top", "Mouth Right"]
#nameOfMovPT = ["Eye Top(mm)", "Eye Bottom", "Mouth Right"]
my_figsize, my_dpi = (20, 10), 80
Z_cam = 500 #(millimeter)
sizeOfLdmk = [68,2]
desiredEyePixels = 150 #(150 pixel = 6cm, => 1 pixel = 0.4 mms)
eyeDistGT = 63.0 # The distance between middle of eyes is 60mm
pix2mm = eyeDistGT/desiredEyePixels

svVideo = 'output_moving1.mov'
cap = cv2.VideoCapture('./dataSample/3_moves_video.mov') #load video
#szLdmkBeforeTrans = 'landmark_before_trans_moving.txt'
#svLdmkAfterTrans = 'landmark_after_trans_moving.txt'


# video info
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
totalFrame = np.int32(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print "Total frames: ", totalFrame
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
vis = 0
fourcc = cv2.cv.CV_FOURCC(*'XVID') # create VideoWriter object
width, height = my_figsize[0] * my_dpi, my_figsize[1] * my_dpi
out = cv2.VideoWriter(svVideo, fourcc, fps, (width, height))

while(cap.isOpened()):
    frameIndex = np.int32(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
    print "Processing frame ", frameIndex, "..."
    # capture frame-by-frame
    ret, frame = cap.read()
    ldmkBeforeTrans = np.empty([totalFrame * sizeOfLdmk[0], sizeOfLdmk[1]]);
    if ret==True:
        # operations on the frame
        (bb, frame_landmarks) = helpers.get_landmarks(frame)
        ldmkBeforeTrans =  np.append(ldmkBeforeTrans, frame_landmarks);
        (eyeLNow,eyeRNow,noseNow) = helpers.get_fixedPoint(frame_landmarks, numOfPoint = 3)
        (vertices, mesh_plotting, Ind) = helpers.landmarks_3d_fitting(frame_landmarks,height,width)
        # normalize
        (norm_frame, norm_landmarks) = helpers.normalize(frame, frame_landmarks, desiredEyePixels)
        # 3D transformation by adding depth to the 2d image
        (vertices, mesh_3d_points, Ind) = helpers.landmarks_3d_fitting(norm_landmarks,height,width)
        frame_landmarks_trans = vertices[np.int32(Ind),0:2]
       
        
        # compare current landmarks with the first frame landmarks
        if frameIndex == 0:
            (eyeL0,eyeR0,nose0) = helpers.get_fixedPoint(norm_landmarks, numOfPoint = 3)
            init_im, init_landmarks = norm_frame, frame_landmarks_trans
            
            diff = frame_landmarks_trans - init_landmarks
            y1 = y2 = y3 = np.zeros(1)
            
        else:
            diff = frame_landmarks_trans - init_landmarks
            y1 = np.append(y1, dist(diff[movPt[0]])) # left inner eye brow
            y2 = np.append(y2, dist(diff[movPt[1]])) # nose
            y3 = np.append(y3, dist(diff[movPt[2]])) # right corner of lips
            
            
    else:
        break
    
    # plotting
    fig = plt.figure(figsize=my_figsize, dpi=my_dpi)
    canvas = FigureCanvas(fig)
    im1 = helpers.visualize_facial_landmarks(frame, bb, frame_landmarks, 1, movPt) # with background
    im2 = helpers.visualize_facial_landmarks(frame, bb, frame_landmarks, 0, movPt) # no background
    # add mesh
    '''
    for (x, y) in mesh_plotting[:,0:2]:
        x = np.int32(x)
        y = np.int32(y)
        cv2.circle(im1, (x,y), 1 , (1, 254, 1), -1)
    '''
    gs = gridspec.GridSpec(6, 3)
    gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[:3, :1])
    ax1.imshow(im1), ax1.set_title('Raw RGB Video', fontsize=16)
    ax1.set_xlabel('Pixel', fontsize=14), ax1.set_ylabel('Pixel', fontsize=14)
    
    ax2 = plt.subplot(gs[:3, 1:2])
    ax2.imshow(im2), ax2.set_title('Landmark Extraction on Raw Data', fontsize=16)
    ax2.set_xlabel('Pixel', fontsize=14), ax2.set_ylabel('Pixel', fontsize=14)
      
    ax3 = plt.subplot(gs[:3, 2:3], projection='3d')
    head_move_box.head_box_plot(ax3,eyeL0*pix2mm,eyeR0*pix2mm,nose0*pix2mm,
        eyeLNow*pix2mm,eyeRNow*pix2mm,noseNow*pix2mm,Z_cam*pix2mm)
    ax3.set_title('Head Movement Tracking', fontsize=16)
    ax3.set_xlabel('mm', fontsize=14), ax3.set_ylabel('mm', fontsize=14), ax3.set_zlabel('mm', fontsize=14)
    
    x=np.arange(frameIndex+1) / fps
    maxMov = 10
    ax_Pt1 = plt.subplot(gs[3, :])
    ax_Pt1.set_title('Landmark Movements (Euclidean Distance)', fontsize=16)
    ax_Pt1.plot(x, y1 * pix2mm,color='cornflowerblue'), ax_Pt1.axis([0, totalFrame/fps, 0, maxMov])
    plt.xlabel("time(s)"), plt.ylabel(nameOfMovPT[0], fontsize=14)
    
    ax_Pt2 = plt.subplot(gs[4, :])
    ax_Pt2.plot(x, y2 * pix2mm,color='navajowhite'), ax_Pt2.axis([0, totalFrame/fps, 0, maxMov])
    plt.xlabel("time(s)", fontsize=14), plt.ylabel(nameOfMovPT[1], fontsize=14)
    
    ax_Pt3 = plt.subplot(gs[5, :])
    ax_Pt3.plot(x, y3 * pix2mm,'m'), ax_Pt3.axis([0, totalFrame/fps, 0, maxMov])
    plt.xlabel("time(s)", fontsize=14), plt.ylabel(nameOfMovPT[2], fontsize=14)
    
    fig.canvas.draw()
    outFrame = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    
    if (vis):
        cv2.imshow('frame',outFrame)
          
    # write the flipped frame
    out.write(outFrame)
    plt.close()


#np.savetxt(szLdmkBeforeTrans, ldmkBeforeTrans) #can also save diff if needed
#np.savetxt(svLdmkAfterTrans, landmarksAllFrame) #can also save diff if needed
cap.release()
out.release()
cv2.destroyAllWindows()
