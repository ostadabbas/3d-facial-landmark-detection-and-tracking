import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import time
import os

import helpers
import head_move_box


def dist(arr):
    # compute distance
    return np.sqrt((arr[0] ** 2) + (arr[1] ** 2))


def tracking(resource, target):
    start = time.time()

    #parameter setting
    movPt = [22,39,57]  #22-leftbrow, 33-righteye, 57-lowerlip
    # nameOfMovPT = ["(6)RightJaw (mm)", "(12)LeftJaw", "(57)LowerLip"]
    nameOfMovPT = ["Left Brow", "Right Eye", "Lower Lip"]
    #nameOfMovPT = ["Euclidean (mm)", "Horizontal", "Vertical"]
    my_figsize, my_dpi = (20, 10), 80
    Z_cam = 500 #(millimeter)
    sizeOfLdmk = [68,2]
    desiredEyePixels = 180 #(180 pixel = 6cm, => 1 pixel = 0.4 mms)
    eyeDistGT = 63.0 # The distance between middle of eyes is 60mm
    pix2mm = eyeDistGT/desiredEyePixels

    svVideo = os.path.join(target, 'output.avi') # create output video file
    sv2DLdMarks = os.path.join(target, '2d_landmarks') # create 2D landmarks file
    sv3DLdMarks = os.path.join(target, '3d_landmarks') # create 3D frontalised landmarks file
    sv3DLdMarks_Pose = os.path.join(target, '3d_landmarks_pose') # create 3D landmarks coupled with pose file
    svNonDetect = os.path.join(target, 'NonDetected') # create non-detected frames file

    landmarks_2d = []
    landmarks_3d = []
    landmarks_pose_3d = []
    nonDetectFr = []

    cap = cv2.VideoCapture(resource)  # load video
    # video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalFrame = np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Total frames: ", totalFrame)
    print("Frame size: ", size)
    vis = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # create VideoWriter object
    width, height = my_figsize[0] * my_dpi, my_figsize[1] * my_dpi
    out = cv2.VideoWriter(svVideo, fourcc, fps, (width, height))
    flagIndx = False
    totalIndx = 0
    while(cap.isOpened()):
        frameIndex = np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("Processing frame ", frameIndex, "...")
        # capture frame-by-frame
        ret, frame = cap.read()
        if ret==True:
            # operations on the frame
            try:
                # generate face bounding box and track 2D landmarks for current frame
                (bb, frame_landmarks) = helpers.get_landmarks(frame)
            except:
                print("Landmarks in frame ", frameIndex, " (", frameIndex/fps, " s) could not be detected.")
                nonDetectFr.append(frameIndex/fps)
                continue

            # only for plotting head movement
            (eyeLNow,eyeRNow,noseNow) = helpers.get_fixedPoint(frame_landmarks, numOfPoint = 3)

            # 3D transformation by adding depth to the 2D image
            (vertices, mesh_plotting, Ind, rotation_angle) = helpers.landmarks_3d_fitting(frame_landmarks,height,width)
            frame_landmarks_3d = vertices[np.int32(Ind),0:2]

            landmarks_2d.append(frame_landmarks)
            landmarks_3d.append(vertices[np.int32(Ind),0:3])
            landmarks_pose_3d.append(mesh_plotting[np.int32(Ind),0:3])

            totalIndx = totalIndx + 1
            # compare current landmarks with the first frame landmarks
            if flagIndx == False:
                eyeL0,eyeR0,nose0 = eyeLNow, eyeRNow, noseNow
                init_im = frame
                init_landmarks = frame_landmarks
                init_landmarks_3d = frame_landmarks_3d

                diff = frame_landmarks - init_landmarks
                # dis for Euclidean distance, dis_x for horizontal displacement, dis_y for vertical displacement
                # dis = dis_x = dis_y = np.zeros(1)

                y1 = y2 = y3 = np.zeros(1)

                flagIndx = True
            else:
                diff = frame_landmarks_3d - init_landmarks_3d
                # dis = np.append(dis, dist(diff[movPt])) # left jaw
                # dis_x = np.append(dis_x, diff[movPt, 0])
                # dis_y = np.append(dis_y, diff[movPt, 1])
                y1 = np.append(y1, dist(diff[movPt[0]]))  # Leftbrow

                y2 = np.append(y2, dist(diff[movPt[1]]))  # Righteye

                y3 = np.append(y3, dist(diff[movPt[2]]))  # Lower lip

        else:
            break

        ############################ plotting ##############################
        fig = plt.figure(figsize=my_figsize, dpi=my_dpi)
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(6, 3)
        gs.update(wspace=0.5)
        gs.update(hspace=1)

        ##################### Raw data with landmarks ######################
        im1 = helpers.visualize_facial_landmarks(frame, bb, frame_landmarks, 1, movPt[0:3])  # with background
        # im2 = helpers.visualize_facial_landmarks(frame, bb, frame_landmarks, 0, movPt)  # no background
        # add mesh
        # for (x, y) in mesh_plotting[:, 0:2]:
        #     x = np.int32(x)
        #     y = np.int32(y)
        #     cv2.circle(im1, (x, y), 1, (1, 254, 1), -1)

        ax1 = plt.subplot(gs[:3, :1])
        ax1.imshow(im1)
        ax1.set_title('Raw RGB Video', fontsize=16)
        ax1.set_ylabel('Pixel', fontsize=14)

        ######################### landmark tracking ########################
        ax2 = plt.subplot(gs[:3, 1:2])
        # ax2.imshow(im2),ax2.set_title('Landmark Extraction on Raw Data', fontsize=16)
        ax2.set_title('Landmarks Tracking', fontsize=16)
        ax2.set_ylabel('Pixel', fontsize=14)
        ax2.axis([-100, 100, -100, 100])
        # landmarkCompare = 0 * im1.copy() + 255
        for (x, y) in frame_landmarks_3d[:, 0:2]:
            x = np.int32(x)
            y = np.int32(y)
            plt.plot(x, y, 'go')

        # for highlight
        (a1, b1) = frame_landmarks_3d[movPt[0], 0:2]  # 22-leftbrow
        (a2, b2) = frame_landmarks_3d[movPt[1], 0:2]  # 33-righteye
        (a3, b3) = frame_landmarks_3d[movPt[2], 0:2]  # 57-lowerlip
        plt.plot(a1, b1, 'o', color = 'cornflowerblue')
        plt.plot(a2, b2, 'o', color = 'navajowhite')
        plt.plot(a3, b3, 'o', color = 'm')

        # for jaw contour
        # plt.plot(frame_landmarks_3d[0:17, 0], frame_landmarks_3d[0:17, 1], linestyle='-', color='r', lw=2)

        ###################### head movement tracking ######################
        ax3 = plt.subplot(gs[:3, 2:3], projection='3d')
        rotationX, rotationY, rotationZ = rotation_angle
        head_move_box.head_box_plot(ax3, eyeL0 * pix2mm, eyeR0 * pix2mm, nose0 * pix2mm,
                                        eyeLNow * pix2mm, eyeRNow * pix2mm, noseNow * pix2mm,
                                        rotationX, rotationY, rotationZ, Z_cam * pix2mm)
        ax3.set_title('Head Movement Tracking', fontsize=16)
        ax3.set_xlabel('mm', fontsize=14), ax3.set_ylabel('mm', fontsize=14), ax3.set_zlabel('mm', fontsize=14)

        ################## landmark movements #################
        x = np.arange(totalIndx) / fps
        maxMov = max(y1 * pix2mm) + 1
        minMov = min(y1 * pix2mm) - 1
        ax_Pt1 = plt.subplot(gs[3, :])
        ax_Pt1.set_title('Movement of 3 Highlight Point ', fontsize=16)
        ax_Pt1.plot(x, y1 * pix2mm, color='cornflowerblue')
        ax_Pt1.axis([0, totalFrame / fps, minMov, maxMov])
        plt.xlabel("time(s)")
        plt.ylabel(nameOfMovPT[0], fontsize=14)

        maxMov = max(y2 * pix2mm) + 1
        minMov = min(y2 * pix2mm) - 1
        ax_Pt2 = plt.subplot(gs[4, :])
        ax_Pt2.plot(x, y2 * pix2mm, color='navajowhite')
        ax_Pt2.axis([0, totalFrame / fps, minMov, maxMov])
        plt.xlabel("time(s)", fontsize=14)
        plt.ylabel(nameOfMovPT[1], fontsize=14)

        maxMov = max(y3 * pix2mm) + 1
        minMov = min(y3 * pix2mm) - 1
        ax_Pt3 = plt.subplot(gs[5, :])
        ax_Pt3.plot(x, y3 * pix2mm, color='m')
        ax_Pt3.axis([0, totalFrame / fps, minMov, maxMov])
        plt.xlabel("time(s)", fontsize=14)
        plt.ylabel(nameOfMovPT[2], fontsize=14)

        fig.canvas.draw()
        outFrame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

        if (vis):
            cv2.imshow('frame', outFrame)

        # write the flipped frame
        out.write(outFrame)
        plt.close()


    np.save(sv3DLdMarks, np.asarray(landmarks_3d))
    np.save(sv2DLdMarks, np.asarray(landmarks_2d))
    np.save(sv3DLdMarks_Pose, np.asarray(landmarks_pose_3d))
    np.save(svNonDetect, np.asarray(nonDetectFr))

    cap.release()
    out.release()

    end = time.time()
    print("processing time:" + str(end - start))

if __name__ == "__main__":
    path = "./videos"
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        print(filepath)
        target = os.path.join(os.getcwd(), os.path.basename(filepath).split('.')[0])
        if not os.path.exists(target):
            os.mkdir(target)
        tracking(filepath, target)
