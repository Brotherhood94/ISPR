
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint


def process_file(load_path, save_path):
    img = cv2.imread(load_path)
    g_img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(100)
    kp, des = sift.detectAndCompute(g_img,None)
    kp_img = cv2.drawKeypoints(g_img,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(save_path, kp_img)


def start():
    dir = './dataset/chosen/'
    saveDir = './dataset/keypoints_chosen/'
    for file in os.listdir(dir):
        if file.endswith(".bmp"):
            process_file(load_path=dir+''+file, save_path=saveDir+''+file)


def compare_keypoints():
    #Load an Image
    img_path_1='./dataset/chosen/2_5_s.bmp'
    img_path_2='./dataset/chosen/6_10_s.bmp'

    img_1 = cv2.imread(img_path_1) 
    g_img_1 = cv2.cvtColor (img_1, cv2.COLOR_BGR2GRAY)

    img_2 = cv2.imread(img_path_2)
    g_img_2 = cv2.cvtColor (img_2, cv2.COLOR_BGR2GRAY)


    sift = cv2.xfeatures2d.SIFT_create(19)

    kp_1, des_1 = sift.detectAndCompute(g_img_1, None)   
    kp_2, des_2 = sift.detectAndCompute(g_img_2, None) 

    kp_img_1 = cv2.drawKeypoints(g_img_1,kp_1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img_2 = cv2.drawKeypoints(g_img_2,kp_2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, ax = plt.subplots()

    index = np.arange(len(des_1[0]))
    bar_width = 0.40
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    ax.bar(index, des_1[18], bar_width,
                alpha=opacity, color='b', error_kw=error_config,
                label='Tree')

    ax.bar(index + bar_width, des_2[18], bar_width,
                alpha=opacity, color='r', error_kw=error_config,
                label='Face')

    ax.set_xlabel('Descriptor Vectors')
    ax.legend()
    fig.tight_layout()
    
    plt.show()
    plt.imshow(kp_img_1)

    plt.show()
    plt.imshow(kp_img_2)
    
    plt.show()
    cv2.imwrite('./kp_img_1_barplot.bmp', kp_img_1)
    cv2.imwrite('./kp_img_2_barplot.bmp', kp_img_2)

def find_patch():
    img_path_detail='./dataset/chosen/waldo_hat.png'
    img_path_entire='./dataset/chosen/waldoscen2.jpg'

    img_detail = cv2.imread(img_path_detail)
    img_detail =  cv2.cvtColor (img_detail, cv2.COLOR_BGR2GRAY)

    img_entire = cv2.imread(img_path_entire, cv2.COLOR_RGB2GRAY) 
    img_entire =  cv2.cvtColor (img_entire, cv2.COLOR_BGR2GRAY)

    #SettingUp SIFT parameters
    nfeatures=100
    nOctaveLayers=20
    contrastThreshold=0.2
    edgeThreshold=0.8
    sigma=20

    sift = cv2.xfeatures2d.SIFT_create()

    kp_detail, des_detail = sift.detectAndCompute(img_detail, None)   
    kp_entire, des_entire = sift.detectAndCompute(img_entire, None) 

    kp_img_detail = cv2.drawKeypoints(img_detail,kp_detail,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_img_entire = cv2.drawKeypoints(img_entire,kp_entire,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    bf = cv2.BFMatcher()
    matches = bf.match(des_detail,des_entire)
    matches = sorted(matches, key = lambda m: m.distance)

    matching_result = cv2.drawMatches(kp_img_detail, kp_detail, kp_img_entire, kp_entire, matches[:100], None)
    cv2.imshow("Matching", matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


start()
#compare_keypoints()
#find_patch()



'''
print(len(keypoints))
print(keypoints[0].pt)
print(keypoints[0].octave)
print(keypoints[0].size)
print(keypoints[0].angle)
'''

#--------------------INFO---------------

#http://www.vlfeat.org/api/sift.html
#http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/
#https://www.theopavlidis.com/technology/CBIR/PaperE/AnSIFT1.htm
#https://stackoverflow.com/questions/48385672/opencv-python-unpack-sift-octave
#https://pythonhosted.org/sift_pyocl/sift.html
#https://stackoverflow.com/questions/46720075/the-value-of-128-sift-descriptor
#https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients#Gradient_computation
#https://stackoverflow.com/questions/39263646/opencv-how-to-calculate-sift-descriptor-at-a-given-pixel
#http://www.silx.org/pub/doc/silx/0.4.0/Tutorials/Sift/sift.html
#https://www.learnopencv.com/histogram-of-oriented-gradients/

# Regarding different parameters, the paper gives some empirical 
# data which can be summarized as, number of octaves = 4,
# number of scale levels = 5,
# initial σ=1.6, k=2–√ etc as optimal values.


'''

An orientation has to be assigned to each keypoint so that SIFT descriptors will be invariant to rotation. For each blurred version of the image,
the gradient magnitude and orientation are computed. From the neighborhood of a keypoint, a histogram of orientations is built (36 bins, 1 bin per 10 degrees).
The drawback is that it is mathematically complicated and computationally heavy.
SIFT is based on the Histogram of Gradients. Thatis, the gradients of each Pixel in the patch need to be computed and these computations cost time.
It is not effective for low powered devices.



    Still quite slow (SURF provides similar performance while running faster)
    Generally doesn't work well with lighting changes and blur

    The  experimental  results  show  that  each  algorithm  has  its  own  advantage.  SIFT  and  CSIFT  perform  the  best
    under  scale and rotation change. CSIFT improves SIFT under blur and  affine  changes,  but  not  illumination  change.  GSIFT  performs  the  best  under  blur  and  illumination  changes.  
    ASIFT performs the best under affine change. PCA-SIFT is always  the  second  in  different  situations.  SURF  performs  th
    e worst in different situations, but runs the fastest.


    If you are a beginner in computer vision, the image in the center is very informative. 
    It shows the patch of the image overlaid with arrows showing the gradient — the arrow 
    shows the direction of gradient and its length shows the magnitude. Notice how the 
    direction of arrows points to the direction of change in intensity and the magnitude 
    shows how big the difference is.

    the angles are between 0 and 180 degrees instead of 0 to 360 degrees.
    But, why not use the 0 – 360 degrees ? Empirically it has been shown that unsigned gradients work better than signed gradients for pedestrian detection.
    Some implementations of HOG will allow you to specify if you want to use signed gradients.

    In our representation, the y-axis is 0 degrees. You can see the histogram has a lot of weight near 0 and 180 degrees, which 
    is just another way of saying that in the patch gradients are pointing either up or down.
'''