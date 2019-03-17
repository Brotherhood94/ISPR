#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#/home/alessandro/Desktop/github/ISPR/Mid1/dataset/chosen/1_2_s.bmp
#Comparare keypoint a colori e gray?

def process_file(load_path, save_path, mask_path = None):
    if(mask_path is not None):
        print("ciao")
        mask_path = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_GRAY2RGB)
    img = cv2.imread(load_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, mask_path) #Maske
    kp, des = sift.compute(gray,kp)
    img = cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #print(kp)
    #plt.plot(des)
    plt.show()
    cv2.imwrite(save_path, img)



dir = '/home/alessandro/Desktop/github/ISPR/Mid1/dataset/chosen/'
saveDir = '/home/alessandro/Desktop/github/ISPR/Mid1/dataset/keypoints_chosen/'
maskDir = '/home/alessandro/Desktop/github/ISPR/Mid1/dataset/chosen/GT/'
for file in os.listdir(dir):
    if file.endswith(".bmp"):
        process_file(load_path=dir+''+file, save_path=saveDir+''+file, mask_path=maskDir+''+file+'_GT')
