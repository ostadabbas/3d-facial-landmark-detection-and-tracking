# Landmark Detection and Tracking

The main objective of this code is to track 68 facial landmarks from each frame of a video.

## Contents   
*  [Requirements](#Requirements)
*  [Preprocessing](#Preprocessing)
*  [Running the Code](#Running the Code)


### Requirements   
Two libraries need to be installed:

* [Dlib 19.10.0](http://dlib.net) for face detection and 2D facial landmarks tracking.
        Use 'pip install dlib==19.10.0' to install this library directly.
* [EOS 1.0.1](https://github.com/patrikhuber/eos/releases) which is a lightweight 3D Morphable Face Model fitting library.
        Use 'pip install eos-py==1.0.1' to install this library directly.

Other requirments are listed in `requirements.txt'.

### Running the Code
Run `landmark_detection_video.py` to get the results.

* ##### input
    some videos as input files, can be put into the `./videos` folder. Of course, the format of video file could not only be `MP4`,
    but also `AVI` or `MOV`.
    
    If we want to change the path of the videos, we can find a function named `main` in the end of `landmark_detection_video.py`,
    and change the path `./videos` to you want. 
* ##### output
    Some folders, named the video files' names, will be generated under root path after completing landmarks tracking. In each
    folder, there are 5 files:
        (1) "output.avi" (synchronized with the original video but filtered non-detected frames)
        (2) "3d_landmarks.npy" (3D frontalised facial landmarks positions over time)
        (3) "3d_landmarks_pose.npy" (3D facial landmarks with head pose over time)
        (4) "2d_landmarks.npy" (2D facial landmarks positions over time)
        (5) "NonDetected" (frames that ficial landmarks cannot be detected)

Run `landmark_detection_img.py` to visualiz detected 68 2D landamarks for images. Please customize code and parameters according to your needs. 

## Citation
If you find our work useful in your research please consider citing our paper:

@INPROCEEDINGS{facial2018yin,

  title     = {Facial Expression and Peripheral Physiology Fusion to Decode Individualized Affective Experience},  
  author    = {Yin, Y and Nabian, M and Fan, M and Chou, C and Gendron, M and Ostadabbas, S},  
  booktitle = {2nd Affective Computing Workshop of the 27th International Joint Conference on Artificial Intelligence (IJCAI)},  
  year      = {2018}  
  
}

## For further inquiry please contact: 
Sarah Ostadabbas, PhD
Electrical & Computer Engineering Department
Northeastern University, Boston, MA 02115
Office Phone: 617-373-4992
ostadabbas@ece.neu.edu
Augmented Cognition Lab (ACLab) Webpage: http://www.northeastern.edu/ostadabbas/
