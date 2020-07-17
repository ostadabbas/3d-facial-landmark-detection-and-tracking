# Landmark Detection and Tracking

The main objective of this code is to track 68 facial landmarks from each frame of a video.

## Contents   
*  [Requirements](#Requirements)
*  [Video Requirements](#Video Requirements)
*  [Running the Code](#Running the Code)
*  [Application](#Application)



### Requirements   
Two libraries need to be installed:

* [Dlib 19.10.0](http://dlib.net) for face detection and 2D facial landmarks tracking.
        Use 'pip install dlib==19.10.0' to install this library directly.
* [EOS 1.0.1](https://github.com/patrikhuber/eos/releases) which is a lightweight 3D Morphable Face Model fitting library.
        Use 'pip install eos-py==1.0.1' to install this library directly.

Other requirments are listed in `requirements.txt'.

### Video Requirements 
We applied a widely used HOG+SVM based face detector to detect faces. It is the fastest method on CPU but struggles to handle non-frontal faces at odd angles. And it may fail to detect small faces as it was trained for minimum face size of 80x80. 

Here are some notes on recording good quality videos：
1. Under good lighting conditions and have a relatively simple background.
2. Try to make sure the camera is facing the frontal face or slightly non-frontal faces.
3. Try to fix camera position to reduce camera's jitters.
4. Keep a certain distance between the camera and the person so that the captured face is not too big or too small.

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

### Application
By leveraging this facial landmarks tracking technology to extract the movement signals of baby's jaw from recorded baby's sucking video, we proposed a novel contact-less data acquisition and quantification scheme for Non-nutritive sucking (NNS), which can be regarded as an indicator of infant's central nervous system development.
More details is in the paper ["Infant Contact-less Non-Nutritive Sucking Pattern Quantification via Facial Gesture Analysis"](https://arxiv.org/pdf/1906.01821.pdf).

## Citation
If you find our work useful in your research please consider citing our papers:

@INPROCEEDINGS{facial2018yin,

  title     = {Facial Expression and Peripheral Physiology Fusion to Decode Individualized Affective Experience},  
  author    = {Yin, Y and Nabian, M and Fan, M and Chou, C and Gendron, M and Ostadabbas, S},  
  booktitle = {2nd Affective Computing Workshop of the 27th International Joint Conference on Artificial Intelligence (IJCAI)},  
  year      = {2018}  
  
}

@article{huang2019infant,
  title={Infant Contact-less Non-Nutritive Sucking Pattern Quantification via Facial Gesture Analysis},
  author={Huang, Xiaofei and Martens, Alaina and Zimmerman, Emily and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:1906.01821},
  year={2019}
}

## For further inquiry please contact: 
Sarah Ostadabbas, PhD
Electrical & Computer Engineering Department
Northeastern University, Boston, MA 02115
Office Phone: 617-373-4992
ostadabbas@ece.neu.edu
Augmented Cognition Lab (ACLab) Webpage: http://www.northeastern.edu/ostadabbas/
