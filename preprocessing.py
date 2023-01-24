# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:07:23 2023

@author: arzu.yildiz
"""

import cv2
import numpy as np
from pytesseract import Output
import pytesseract
import imutils
from PIL import Image


img = cv2.imread('data/yazi90.png')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    print(coords)
    print(cv2.minAreaRect(coords))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def ocrImage(image):
    print(pytesseract.image_to_string(image, lang='tur'))
    
def orientation():
    
    image = cv2.imread("data/yazi90.png")
    ocrImage(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
    
    # display the orientation information
    print("[INFO] detected orientation: {}".format(
    	results["orientation"]))
    print("[INFO] rotate by {} degrees to correct".format(
    	results["rotate"]))
    print("[INFO] detected script: {}".format(results["script"]))
    
    
    # rotate the image to correct the orientation
    rotated = imutils.rotate_bound(image, angle=results["rotate"])
    # show the original image and output image after orientation
    # correction
    ocrImage(rotated)
    cv2.imshow("Original", image)
    cv2.imshow("Output", rotated)
    cv2.waitKey(0)
    
    
    
orientation()