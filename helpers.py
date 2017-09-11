#!/usr/bin/python


import dlib
import numpy as np
from collections import OrderedDict
import cv2
import eos

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
    ("jaw", (0, 17))
])

model = eos.morphablemodel.load_model("./share/sfm_shape_3448.bin")
blendshapes = eos.morphablemodel.load_blendshapes("./share/expression_blendshapes_3448.bin")
landmark_mapper = eos.core.LandmarkMapper('./share/ibug_to_sfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology('./share/sfm_3448_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load('./share/ibug_to_sfm.txt')
model_contour = eos.fitting.ModelContour.load('./share/model_contours.json')
landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
    
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    #print dir(shape)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords
    
def get_landmarks(im):
        predictor_path = "./shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        rects = detector(im, 1)
        for k, d in enumerate(rects):
                shape = predictor(im, d)
                
                corrds = shape_to_np(shape)
                bb = rect_to_bb(d)
                return (bb, corrds)

def visualize_facial_landmarks(image, bb, shape, background=1, highlightPt=[]):
        # background==1: show video background (0 means showing only landmarks, no background)
        color=[(86,158,248), (248, 221, 187), (255,0,255)] #(252, 47, 7)#(200, 156, 30),(100, 254, 253),(174, 210, 253)
        if (background):
            overlay = image.copy()
        else:
            overlay = 0 * image.copy()# + 255
        
        cv2.rectangle(overlay, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (199, 204, 248), 2)
        for (x, y) in shape:
            overlay[y,x] = (0,0,0)
            cv2.circle(overlay, (x,y), 2 , (0, 254, 0), -1)
        
        for (i, (x, y)) in enumerate(shape[highlightPt]):  # Highlight
            overlay[y,x] = (0,0,0)
            cv2.circle(overlay, (x,y), 8 , color[i], -1)
        return overlay
            
def get_fixedPoint(shape, numOfPoint = 3):
    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]
    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    
    if numOfPoint == 3:

        nosePt = shape[33].astype("int")
        
        outputs = np.concatenate((leftEyeCenter, rightEyeCenter, nosePt), axis=0).reshape(3, 2)
    else:
        Pt1 = shape[5].astype("int")
        Pt2 = shape[11].astype("int")
        #Pt3 = shape[36].astype("int")
        #Pt4 = shape[42].astype("int")
        
        #outputs = np.concatenate((Pt1, Pt2, Pt3, Pt4), axis=0).reshape(4, 2)
        outputs = np.concatenate((leftEyeCenter, rightEyeCenter, Pt1, Pt2), axis=0).reshape(4, 2)

    return (outputs)

def geoTrans(init_im, init_landmarks, src_im, src_landmarks):
    rows2,cols2,ch2 = src_im.shape
    
    ################### affine transformation ##########################
    pts1 = np.float32(get_fixedPoint(init_landmarks,3))
    pts2 = np.float32(get_fixedPoint(src_landmarks,3))
    #print "fixed point in initial image:", "\n", pts1, "\n"
    #print "fixed point in test image:", "\n", pts2, "\n"

    M = cv2.getAffineTransform(pts2,pts1)
    #print "Transform matrix:\n", M, "\n"

    dst = cv2.warpAffine(src_im,M,(cols2,rows2))
    dst_landmarks = np.concatenate((np.transpose(src_landmarks), np.ones(src_landmarks.shape[0]).reshape(1,68)), axis=0) 
    dst_landmarks = np.dot(M, dst_landmarks).T
    dst_landmarks = np.int32(dst_landmarks)
    return (dst, dst_landmarks)
    
    '''
    dst = src_im
    dst_landmarks = src_landmarks
    
    ################### pers transformation ##########################
    pts3 = np.float32(get_fixedPoint(init_landmarks,4))
    pts4 = np.float32(get_fixedPoint(dst_landmarks,4))
    #print "fixed point in initial image:", "\n", pts3, "\n"
    #print "fixed point in test image:", "\n", pts4, "\n"

    M2 = cv2.getPerspectiveTransform(pts4,pts3)
    #print "Transform matrix:\n", M2, "\n"

    dst2 = cv2.warpPerspective(dst, M2, (cols2,rows2))  

    dst_landmarks2 = np.concatenate((np.transpose(dst_landmarks), np.ones(dst_landmarks.shape[0]).reshape(1,68)), axis=0) 
    dst_landmarks2 = np.dot(M2, dst_landmarks2)
    dst_landmarks2 = np.divide(dst_landmarks2[:2, :], dst_landmarks2[2, :]).T
    dst_landmarks2 = np.int32(dst_landmarks2)
    
    return (dst2, dst_landmarks2)
    '''

def normalize(image, shape,  desiredEyePixels):
    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]
    
    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    scale = desiredEyePixels / dist
    
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    
    # apply the affine transformation
    rows,cols = image.shape[:2]
    output_image = cv2.warpAffine(image, M, (cols, rows))
    
    dst_landmarks = np.concatenate((np.transpose(shape), np.ones(shape.shape[0]).reshape(1,68)), axis=0) 
    dst_landmarks = np.dot(M, dst_landmarks).T
    dst_landmarks = np.int32(dst_landmarks)
    
    # return the aligned face
    return (output_image, dst_landmarks)
    
def landmarks_3d_fitting(landmarks,image_height, image_width):
    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
            landmarks, landmark_ids, landmark_mapper,
            image_width, image_height, edge_topology, contour_landmarks, model_contour)

    vertices = np.array(mesh.vertices)
    #texcoords = np.array(mesh.texcoords)
    #tvi = np.array(mesh.tvi)
    #print pose.get_rotation() # (4, 1)
    #print pose.get_projection() # (4, 4)
    #print pose.get_modelview() # (4, 4)
    #print pose.get_rotation_euler_angles() # (3, 1)
    w2, h2 = image_width/2, image_height/2
    viewport = np.array([[w2, 0, 0, w2],
                        [0, h2*(-1), 0, h2],
                        [0, 0, 0.5, 0.5],
                        [0, 0, 0, 1]])
    a = multiplyABC(viewport, pose.get_projection() ,pose.get_modelview())
    a = a.transpose()
    mesh_3d_points = np.dot(vertices, a)
    # landmark index in mesh
    Ind = np.zeros((68,))
    for (i, (x, y)) in enumerate(landmarks):
        Ind[i] = np.argmin((np.square(x-mesh_3d_points[:, 0]) + np.square(y - mesh_3d_points[:,1])))
    
    return (vertices, mesh_3d_points, Ind)

def multiplyABC(A, B, C):
    temp = np.dot(A, B);
    return np.dot(temp, C);