import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('images/IMG_4054.JPG',0)
imgR = cv2.imread('images/IMG_4055.JPG',0)
 
stereo = cv2.StereoBM()
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'red')
plt.show()