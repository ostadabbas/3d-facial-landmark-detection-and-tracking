def plot_cube(ax3,Xc, Yc, Zc, b, big_b):
    # Xc,Yc,Zc are the coordinates of the center of the cube
    # b is the half size of the cube     big_b is half size of entire plot region
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    ax = ax3
    #ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-big_b, big_b])
    ax.set_ylim([-big_b, big_b])
    ax.set_zlim([-big_b, big_b])

    ax.plot([Xc - b, Xc + b], [Yc - b, Yc - b], zs=[Zc - b, Zc - b], color='b')
    ax.plot([Xc - b, Xc + b], [Yc + b, Yc + b], zs=[Zc - b, Zc - b], color='b')
    ax.plot([Xc - b, Xc + b], [Yc - b, Yc - b], zs=[Zc + b, Zc + b], color='b')
    ax.plot([Xc - b, Xc + b], [Yc + b, Yc + b], zs=[Zc + b, Zc + b], color='b')

    ax.plot([Xc - b, Xc - b], [Yc - b, Yc + b], zs=[Zc - b, Zc - b], color='b')
    ax.plot([Xc - b, Xc - b], [Yc - b, Yc + b], zs=[Zc + b, Zc + b], color='b')
    ax.plot([Xc + b, Xc + b], [Yc - b, Yc + b], zs=[Zc - b, Zc - b], color='b')
    ax.plot([Xc + b, Xc + b], [Yc - b, Yc + b], zs=[Zc + b, Zc + b], color='b')

    ax.plot([Xc - b, Xc - b], [Yc - b, Yc - b], zs=[Zc - b, Zc + b], color='b')
    ax.plot([Xc - b, Xc - b], [Yc + b, Yc + b], zs=[Zc - b, Zc + b], color='b')
    ax.plot([Xc + b, Xc + b], [Yc - b, Yc - b], zs=[Zc - b, Zc + b], color='b')
    ax.plot([Xc + b, Xc + b], [Yc + b, Yc + b], zs=[Zc - b, Zc + b], color='b')

    #plt.show()


def x_face_move(X_nose_pixel_0,X_nose_pixel_now):
    X_now=X_nose_pixel_now-X_nose_pixel_0
    return (X_now)

def y_face_move(Y_nose_pixel_0,Y_nose_pixel_now):
    Y_now=Y_nose_pixel_now-Y_nose_pixel_0
    return (Y_now)

def z_face_move(Z_cam,eye_left_pixel_0,eye_right_pixel_0,eye_left_pixel_now,eye_right_pixel_now):
    import math
    miu_0= math.fabs(eye_left_pixel_0-eye_right_pixel_0)
    miu_now=math.fabs(eye_left_pixel_now-eye_right_pixel_now)
    Z_now=(1-(miu_0/miu_now))*Z_cam
    return (-Z_now)

def head_box_plot(ax3,eyeL0,eyeR0,nose0,eyeLNow,eyeRNow,noseNow,Z_cam):
    #Z_cam is the distance from camera to object, we can assume 500 (milimeter)       
    import  math
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
    plot_cube(ax3,X_now, Y_now, Z_now, box_dim/2, box_dim)


# Example :

#head_box_plot(10,20,15,10,
#                  20,30,25,10,
#                  500)





