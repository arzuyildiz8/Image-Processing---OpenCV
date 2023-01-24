# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:01:08 2023

@author: arzu.yildiz
"""

import cv2 
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

def display(image,title):
    plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="gray")
    
def grayscale(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def ocr_pytesseract(image):
    text=pytesseract.image_to_string(image, lang="tur")
    return text


def get_psm(image,config_parameter):
    
    if config_parameter =="--psm 0":
        text = pytesseract.image_to_osd(image)
        
    else:
        text = pytesseract.image_to_string(image, config = config_parameter, lang="tur")
    
    print("[INFO "+str(config_parameter)+"]\n------------")
    print(text)
    
def psm():
    """Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific."""
    
    img0=grayscale("data/yazi90.png")
    psm0= get_psm(img0,"--psm 0")
    
    img1=grayscale("data/yazi90.png")
    psm1= get_psm(img1,"--psm 1")
    
    img3=grayscale("data/yazi90.png")
    psm3= get_psm(img3,"--psm 3")
    
    img4=grayscale("data/test006.JPG")
    psm4= get_psm(img4,"--psm 4")
    
    img5=grayscale("data/test007.JPG")
    psm5= get_psm(img5,"--psm 5")
    
    img6=grayscale("data/test008.JPG")
    psm6= get_psm(img6,"--psm 6")
    
    img7=grayscale("data/test009.JPG")
    psm7= get_psm(img7,"--psm 7")
    
    img8=grayscale("data/test010.JPG")
    psm8= get_psm(img8,"--psm 8")
    
    img9=grayscale("data/test011.JPG")
    psm9= get_psm(img9,"--psm 9")
    
    img10=grayscale("data/test012.JPG")
    psm10= get_psm(img10,"--psm 10")
    
    img11=grayscale("data/test013.JPG")
    psm11= get_psm(img11,"--psm 11")
    
    img12=grayscale("data/test013.JPG")
    psm12= get_psm(img12,"--psm 12")
    
    img13=grayscale("data/test014.JPG")
    psm13= get_psm(img13,"--psm 13")

def noise_removal(image):
    
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

def make_border(image):
    color = [255, 255, 255]
    top, bottom, left, right = [150]*4
    image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_with_border


      
psm()

gray_image=grayscale("data/yazi90.png")

ocr_pytesseract= ocr_pytesseract(gray_image)
print(ocr_pytesseract)


thresh, im_bw = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_image.jpg", im_bw)

no_noise = noise_removal(im_bw)
cv2.imwrite("temp/no_noise.jpg", no_noise)

eroded_image = thin_font(no_noise)
cv2.imwrite("temp/eroded_image.jpg", eroded_image)


dilated_image = thick_font(no_noise)
cv2.imwrite("temp/dilated_image.jpg", dilated_image)


no_borders = remove_borders(no_noise)
cv2.imwrite("temp/no_borders.jpg", no_borders)

image_with_border=make_border(no_borders)
cv2.imwrite("temp/image_with_border.jpg", image_with_border)

