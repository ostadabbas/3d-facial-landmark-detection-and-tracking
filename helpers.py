import dlib
import numpy as np
from collections import OrderedDict
import cv2
import eos
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

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
morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())
landmark_mapper = eos.core.LandmarkMapper('./share/ibug_to_sfm.txt')
edge_topology = eos.morphablemodel.load_edge_topology('./share/sfm_3448_edge_topology.json')
contour_landmarks = eos.fitting.ContourLandmarks.load('./share/ibug_to_sfm.txt')
model_contour = eos.fitting.ModelContour.load('./share/sfm_model_contours.json')
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
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def get_landmarks(im):
    # detect face and track 2D landmarks
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    # define a face detector via Dlib
    detector = dlib.get_frontal_face_detector()
    # define a 2D landmarks tracker via Dlib
    predictor = dlib.shape_predictor(predictor_path)
    rects = detector(im, 1)
    for k, d in enumerate(rects):
        shape = predictor(im, d)
        corrds = shape_to_np(shape)
        bb = rect_to_bb(d)
        return (bb, corrds)


def visualize_facial_landmarks(image, bb, shape, background=1, highlightPt=[]):
    # background==1: show video background (0 means showing only landmarks, no background)
    color=[(86,158,248), (248, 221, 187), (255,0,255)]
    if (background):
        overlay = image.copy()
    else:
        overlay = 0 * image.copy() + 255 # white background
        overlay = 0 * image.copy() # + 255  # black background
    size = overlay.shape
    # cv2.rectangle(overlay, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (199, 204, 248), 2)  # face bounding box

    for (x, y) in shape:  # 68 landmarks
        # print(x,'and', y)
        if(x >= size[1] or y >= size[0]):
            continue
        overlay[y,x] = (0,0,0)
        cv2.circle(overlay, (x,y), 2 , (0, 254, 0), -1)

    # for idx in range(16):  # Jaw's contour
    #    x1 = shape[idx,0]
    #    y1 = shape[idx,1]
    #    x2 = shape[idx+1,0]
    #    y2 = shape[idx+1,1]
    #    cv2.line(overlay, (x1,y1), (x2,y2), (0, 255, 0), 4)

    # Highlight point
        for i in range(len(highlightPt)):
            (x, y) = shape[highlightPt[i]]
            overlay[y, x] = (0, 0, 0)
            cv2.circle(overlay, (x, y), 8, color[i], -1)

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
        outputs = np.concatenate((leftEyeCenter, rightEyeCenter, nosePt), axis=0).reshape(3, -1)
    else:
        Pt1 = shape[5].astype("int")
        Pt2 = shape[11].astype("int")
        #Pt3 = shape[36].astype("int")
        #Pt4 = shape[42].astype("int")

        #outputs = np.concatenate((Pt1, Pt2, Pt3, Pt4), axis=0).reshape(4, 2)
        outputs = np.concatenate((leftEyeCenter, rightEyeCenter, Pt1, Pt2), axis=0).reshape(4, -1)
    return (outputs)


def landmarks_3d_fitting(landmarks,image_height, image_width):
    eos_landmarks = []
    for idx in range(0,68):
        eos_landmarks.append(eos.core.Landmark(str(idx+1), [float(landmarks[idx,0]), float(landmarks[idx,1])]))
    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
        eos_landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour)

    vertices = np.array(mesh.vertices)
    vertices = np.append(vertices, np.ones((vertices.shape[0], 1)), 1)

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
        Ind[i] = np.argmin((np.square(x - mesh_3d_points[:, 0]) + np.square(y - mesh_3d_points[:,1])))
    rotation_angle = pose.get_rotation_euler_angles()
    return (vertices, mesh_3d_points, Ind, rotation_angle)


def multiplyABC(A, B, C):
    temp = np.dot(A, B)
    return np.dot(temp, C)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def MaxDist(X0, Y0, X1, Y1):
    # original X, Y:
    # polynomial curve fit the data
    ori_fittedParameters = np.polyfit(X0, Y0, 3)
    # create data for the fitted equation plot
    ori_xCurve = np.linspace(min(X0), max(Y0), 1000)
    ori_yCurve = np.polyval(ori_fittedParameters, ori_xCurve)

    # current X,Y:
    # polynomial curve fit the data
    fittedParameters = np.polyfit(X1, Y1, 3)
    # create data for the fitted equation plot
    xCurve = np.linspace(min(X1), max(Y1), 1000)
    yCurve = np.polyval(fittedParameters, xCurve)
    # polynomial derivative from numpy
    deriv = np.polyder(fittedParameters)

    maxDist = 0
    point = min(X1)
    step = 1
    xPoint = min(X1)
    while(xPoint <= max(X1)):
        yPoint = np.polyval(fittedParameters, xPoint)
        slope = np.polyval(deriv, xPoint)

        # construct normal line of curve
        x = np.linspace(min(ori_xCurve), max(ori_xCurve), 1000)
        y = (x - xPoint) * (-1.0 / slope) + yPoint

        idx = np.argwhere(np.diff(np.sign(ori_yCurve - y))).flatten()
        xPoint1 = ori_xCurve[idx]
        yPoint1 = ori_yCurve[idx]
        dist = np.sqrt((xPoint - xPoint1)**2 + (yPoint - yPoint1)**2)
        if dist > maxDist:
            maxDist = dist
            point = xPoint

        xPoint = xPoint + 1
    return maxDist, point


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def findSucks(signal, fps):
    # find suck cycles, burst, and compute frequency

    # set parameters of pattern detector
    thres = 0.2  # threshold displacement
    inter = 0.5  # minimal time interval of two continuous cycles
    size = 5  # more than 5 cycles are considered a burst
    peaks, _ = find_peaks(signal, height=thres, distance=10)

    arr = []
    list = []
    t0 = peaks[0] / fps
    for i in range(len(peaks)):
        t = peaks[i] / fps
        if t - t0 <= inter:
            list.append(peaks[i])
        else:
            if len(list) >= size:
                arr.append(list)
            list = [peaks[i]]
        if i == len(peaks) - 1 and len(list) >= size:
            arr.append(list)
        t0 = t

    x = [0.0]
    y = [0]
    widths = peak_widths(signal, peaks, rel_height=0.3)
    paras = []
    for m in range(len(arr)):
        startT = widths[2][peaks == arr[m][0]] / fps
        endT = widths[3][peaks == arr[m][-1]] / fps
        paras.append([startT, endT, len(arr[m])])
        x.extend((startT, startT, endT, endT))
        y.extend((0, 1, 1, 0))

    x.append(len(signal) / fps)
    y.append(0)

    # plot filtered signal and corresponding NNS pattern curve
    t = np.arange(len(signal)) / fps
    plt.plot(t, signal)

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            plt.plot(arr[i][j] / fps, signal[arr[i][j]], "rx")
    plt.plot(np.zeros_like(x), "--", color="gray")

    maxMov = 2
    minMov = -2
    plt.plot(x, y)
    plt.axis([0, len(signal) / fps, minMov, maxMov])

    plt.minorticks_on()
    plt.grid(True)

    # compute average NNS frequency and print result
    totalT = 0.0
    totalCyc = 0
    print("Burst  StartT   EndT   Cycles")
    for burst in range(len(paras)):
        print("%1d      %5.3f    %5.3f    %2d" % (burst + 1, paras[burst][0], paras[burst][1], paras[burst][2]))
        totalT = totalT + (paras[burst][1] - paras[burst][0])
        totalCyc = totalCyc + paras[burst][2]
    freq = totalCyc / totalT
    print("Average suck frequency (Hz): ", freq[0])

    plt.show()
