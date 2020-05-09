import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_cube(ax3, Xc, Yc, Zc, Xr, Yr, Zr, b, big_b):
    # Xc, Yc, Zc are the coordinates of the center of the cube
    # Xr, Yr, Zr are the rotation angel of cube
    # b is the half size of the cube     big_b is half size of entire plot region

    # w=math.pi/180*degree
    # cosw=math.cos(w)
    # sinw=math.sin(w)

    # fig = plt.figure()
    ax = ax3
    # ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-big_b, big_b])
    ax.set_ylim([-big_b, big_b])
    ax.set_zlim([-big_b, big_b])

    p1 = [Xc - b, Yc - b, Zc - b]
    p2 = [Xc + b, Yc - b, Zc - b]
    p3 = [Xc - b, Yc + b, Zc - b]
    p4 = [Xc + b, Yc + b, Zc - b]

    p5 = [Xc - b, Yc - b, Zc + b]
    p6 = [Xc + b, Yc - b, Zc + b]
    p7 = [Xc - b, Yc + b, Zc + b]
    p8 = [Xc + b, Yc + b, Zc + b]

    r11 = math.cos(Xr) * math.cos(Yr)
    r12 = math.cos(Xr) * math.sin(Yr) * math.sin(Zr) - math.sin(Xr) * math.cos(Zr)
    r13 = math.cos(Xr) * math.sin(Yr) * math.cos(Zr) + math.sin(Xr) * math.sin(Zr)

    r21 = math.sin(Xr) * math.cos(Yr)
    r22 = math.sin(Xr) * math.sin(Yr) * math.sin(Zr) + math.cos(Xr) * math.cos(Zr)
    r23 = math.sin(Xr) * math.sin(Yr) * math.cos(Zr) - math.cos(Xr) * math.sin(Zr)

    r31 = -1 * math.sin(Yr)
    r32 = math.cos(Yr) * math.sin(Zr)
    r33 = math.cos(Yr) * math.cos(Zr)

    R = [[r11, r12, r13],
         [r21, r22, r23],
         [r31, r32, r33]]

    p1 = np.matmul(p1, R)
    p2 = np.matmul(p2, R)
    p3 = np.matmul(p3, R)
    p4 = np.matmul(p4, R)
    p5 = np.matmul(p5, R)
    p6 = np.matmul(p6, R)
    p7 = np.matmul(p7, R)
    p8 = np.matmul(p8, R)

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zs=[p1[2], p2[2]], color='b')
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], zs=[p3[2], p4[2]], color='b', linestyle='dashed')
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]], zs=[p5[2], p6[2]], color='b')
    ax.plot([p7[0], p8[0]], [p7[1], p8[1]], zs=[p7[2], p8[2]], color='b')
    ax.plot([p1[0], p3[0]], [p1[1], p3[1]], zs=[p1[2], p3[2]], color='b', linestyle='dashed')
    ax.plot([p2[0], p4[0]], [p2[1], p4[1]], zs=[p2[2], p4[2]], color='b')
    ax.plot([p5[0], p7[0]], [p5[1], p7[1]], zs=[p5[2], p7[2]], color='b')
    ax.plot([p6[0], p8[0]], [p6[1], p8[1]], zs=[p6[2], p8[2]], color='b')
    ax.plot([p1[0], p5[0]], [p1[1], p5[1]], zs=[p1[2], p5[2]], color='b')
    ax.plot([p2[0], p6[0]], [p2[1], p6[1]], zs=[p2[2], p6[2]], color='b')
    ax.plot([p3[0], p7[0]], [p3[1], p7[1]], zs=[p3[2], p7[2]], color='b', linestyle='dashed')
    ax.plot([p4[0], p8[0]], [p4[1], p8[1]], zs=[p4[2], p8[2]], color='b')

    x = [p1[0], p2[0], p6[0], p5[0]]
    y = [p1[1], p2[1], p6[1], p5[1]]
    z = [p1[2], p2[2], p6[2], p5[2]]
    verts = [list(zip(x, y, z))]
    ax.add_collection3d(Poly3DCollection(verts))

    #plt.show()


def x_face_move(X_nose_pixel_0,X_nose_pixel_now):
    X_now=X_nose_pixel_now-X_nose_pixel_0
    return (X_now)

def y_face_move(Y_nose_pixel_0,Y_nose_pixel_now):
    Y_now=Y_nose_pixel_now-Y_nose_pixel_0
    return (Y_now)

def z_face_move(Z_cam,eye_left_pixel_0,eye_right_pixel_0,eye_left_pixel_now,eye_right_pixel_now):
    miu_0= math.fabs(eye_left_pixel_0-eye_right_pixel_0)
    miu_now=math.fabs(eye_left_pixel_now-eye_right_pixel_now)
    Z_now=(1-(miu_0/miu_now))*Z_cam
    return (-Z_now)

def head_box_plot(ax3,eyeL0,eyeR0,nose0,eyeLNow,eyeRNow,noseNow,rotationX,rotationY,rotationZ,Z_cam):
    #Z_cam is the distance from camera to object, we can assume 500 (milimeter)
    eye_X_left_pixel_0 = eyeL0[0]
    eye_X_right_pixel_0 = eyeR0[0]
    (X_nose_pixel_0,Y_nose_pixel_0) = nose0
    eye_X_left_pixel_now = eyeLNow[0]
    eye_X_right_pixel_now = eyeRNow[0]
    (X_nose_pixel_now,Y_nose_pixel_now) = noseNow
    
    X_now=x_face_move(X_nose_pixel_0,X_nose_pixel_now)
    Y_now=y_face_move(Y_nose_pixel_0,Y_nose_pixel_now)
    Z_now=z_face_move(Z_cam,eye_X_left_pixel_0,eye_X_right_pixel_0,eye_X_left_pixel_now,eye_X_right_pixel_now)
    # box_dim= math.fabs(eye_X_left_pixel_0-eye_X_right_pixel_0)*3
    box_dim=180 #Z_cam/2     #How big the dimensions of the plot are
    plot_cube(ax3, X_now, Y_now, Z_now, -rotationY, rotationZ, -rotationX, box_dim/2, box_dim)


# Example :

#head_box_plot(10,20,15,10,
#                  20,30,25,10,
#                  500)

#ax3 = plt.subplot(111,projection='3d')
#plot_cube(ax3,10,10,10,  0.5,0.6,0.9, 20,50)




