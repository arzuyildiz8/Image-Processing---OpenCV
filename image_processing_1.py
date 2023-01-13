# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:15:02 2023

@author: arzu.yildiz
"""
""" Image: Read, Write
"""
import cv2, time

import numpy as np

import matplotlib.pyplot as plt

""" ###### VIDEO ######"""
def readVideo():

    cap = cv2.VideoCapture("video.mp4") # read video
    print("Genişlik: ",cap.get(3)) 
    print("Yükseklik: ", cap.get(4))
    
    if cap.isOpened() == False:
        print("Error")
        
        
    while True:
        ret, frame = cap.read()
        
        frame = cv2.resize(frame, (960, 540))
        
        if ret == True:
            
            time.sleep(0.01) # bunu kullanmazsak video çok hızlı gider.
            
            cv2.imshow("Video",frame)
            
        else: break
        
        # q tuşu ile video kapatılabilir
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release() # stop capture
    cv2.destroyAllWindows() 

#readVideo()

def readVideoWithCamera():
    
    # capture
    cap = cv2.VideoCapture(0) # zero = default camera
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    
    """
    cv.VideoWriter(filename, fourcc, fps, frameSize)
        dosyaadı: Video dosyası girin
        fourcc: çerçeveleri sıkıştırmak için kullanılan 4 karakterli codec kodu
        fps: video akışının kare hızı
        çerçeve boyutu: Çerçevenin yüksekliği ve genişliği
    
    """
    
    # save video
    # WINDOW = DIVX
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    writer = cv2.VideoWriter("video_kaydı.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 20, (width,height))
    
    while True:
        
        ret, frame = cap.read() 
        writer.write(frame)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release() # stop capture
    writer.release()
    cv2.destroyAllWindows()

#readVideoWithCamera()


""" ###### IMAGE ######"""

def importImage():
    
    img = cv2.imread('img1.JPG')
    # Load an color image in grayscale
    img_gray = cv2.imread('img1.JPG',0)
    
    # oku
    cv2.imshow('image',img)
    cv2.imshow('image gray',img_gray)
    
    # cv2.waitKey () bir klavye bağlama işlevidir.
    # İşlev, herhangi bir klavye olayı için belirtilen milisaniye kadar bekler.
    k = cv2.waitKey(0) & 0xFF 
    
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows() # yarattığımız tüm pencereleri yok eder.
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messi_gray.png',img)
        cv2.destroyAllWindows()
        
#importImage()


def resizeAndCropImage():
    # resize and cropping
    img = cv2.imread("img2.JPG")
    print("Resim boyutu: ", img.shape)
    cv2.imshow("Orjinal", img)
    
    imgResized = cv2.resize(img,(1024,1024))
    print("Yeniden boyutlandırılan resim boyutu: ",imgResized.shape)
    cv2.imshow("Image Resized", imgResized)
    
    # Crop
    imgCropped = img[0:400,0:400] 
    cv2.imshow("Kırpılan resim boyutu: ", imgCropped)

#resizeAndCropImage()

def shapeAndTextImage():
    # shape and text
    img = np.zeros((512,512,3),np.uint8) # siyah bir resim oluştur
    print(img.shape)
    cv2.imshow("Image", img)
    
    cv2.line(img,(0,0),(512,512), (0,255,0), 3) # start - stop - color - thickness
    cv2.imshow("Image", img)
    
    cv2.rectangle(img,(0,0),(256,256), (255,0,0), cv2.FILLED) # (bgr)
    cv2.imshow("Image", img)
    
    cv2.circle(img,(300,300),30, (0,0,255), cv2.FILLED) # center - radius
    cv2.imshow("Image", img)
    
    cv2.putText(img,"CIRCLE ",(325,305),cv2.FONT_HERSHEY_COMPLEX, 1 ,(255,0,255))
    cv2.imshow("Image", img)

#╬shapeAndTextImage()

def joiningImage():
    # joining images
    img1 = cv2.imread("img2.JPG")
    img2 = cv2.imread("img1.JPG")
    
    imgResized1 = cv2.resize(img1,(300,300))
    imgResized2 = cv2.resize(img2,(300,300))
    
    cv2.imshow("Orjinal", img1)
    cv2.imshow("Orjinal", img2)
    
    hor = np.hstack((imgResized1,imgResized2))
    cv2.imshow("Horizontal Image", hor)
     
    ver = np.vstack((imgResized1,imgResized2))
    cv2.imshow("Vertical Image", ver)

#joiningImage()

def wrapPerspective():
    # Perspektif Çarpıtma
    img = cv2.imread("kart.png")
    
    width = 400#371
    height = 500 #517
    
    pts1 = np.float32([[203,1],[1,472],[540,150],[338,617]]) # left hand corner
    pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]]) # left hand corner
    
    # perspektif transform matrisi 	3×3 transformation matrix.
    matrix = cv2.getPerspectiveTransform(pts1,pts2) 
    print(matrix)
    
    # 	(width,height) = size of the output image.
    imgOutput = cv2.warpPerspective(img, matrix, (width,height)) # 	(width,height) = size of the output image.
    
    
    cv2.imshow("Image", img)
    cv2.imshow("Warp Image", imgOutput)
    
#wrapPerspective()

def blendingImage():
    # blending
    img1 = cv2.imread("img1.jpg")
    img1= cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("img2.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    plt.figure()
    plt.imshow(img1)
    
    plt.figure()
    plt.imshow(img2)
    
    print(img1.shape)
    print(img2.shape)
    
    img1 = cv2.resize(img1,(600,600))
    print(img1.shape)
    
    img2 = cv2.resize(img2,(600,600))
    print(img1.shape)
    
    plt.figure()
    plt.imshow(img1)
    
    plt.figure()
    plt.imshow(img2)
    
    # blended = alpha*img1 + beta*img2 + gamma
    blended = cv2.addWeighted(src1=img1, alpha = 0.3, src2=img2,beta=0.7, gamma = 0)
    plt.figure()
    plt.imshow(blended)
    
#blendingImage()

def threshing():
    # image thresholding: convering color image to binary
    img = cv2.imread("img1.jpg")
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Orjinal Resim")
    # eşik değeri belirle
    """
    Parametreler
          src girdi dizisi (çok kanallı, 8 bit veya 32 bit kayan nokta).
          dst çıktı dizisi aynı boyut ve tipte ve src ile aynı sayıda kanala sahip.
          eşik değeri.
          THRESH_BINARY ve THRESH_BINARY_INV eşik türleriyle kullanılacak maksimum maksimum değer.
          tür eşikleme türü (bkz. ThresholdTypes).
    """
    # threshold değeri üzerindekileri beyaz yap altındakileri siyah yap
    _, thresh_img = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY)
    
    plt.figure()
    plt.imshow(thresh_img, cmap="gray")
    plt.axis("off")
    plt.title("Threshold")
    """
    Parametreler
        src Kaynak 8 bitlik tek kanallı görüntü.
        dst Aynı boyutta ve src ile aynı tipte hedef görüntüsü.
        maxValue Koşulun karşılandığı piksellere atanan sıfır olmayan değer
        adaptiveMethod Kullanılacak uyarlanabilir eşikleme algoritması, bkz. AdaptiveThresholdTypes. BORDER_REPLICATE | BORDER_ISOLATED, sınırları işlemek için kullanılır.
        THRESH_BINARY veya THRESH_BINARY_INV olması gereken eşik türü Eşikleme türü, bkz. Eşik Türleri.
        blockSize Piksel için bir eşik değeri hesaplamak için kullanılan bir piksel mahallesinin boyutu: 3, 5, 7, vb.
        C Sabit, ortalamadan veya ağırlıklı ortalamadan çıkarılır (aşağıdaki ayrıntılara bakın). Normalde pozitiftir ancak sıfır veya negatif de olabilir.
    """
    """
    Önceki bölümde eşik değer olarak global bir değer kullandık. 
    Ancak görüntünün farklı alanlarda farklı aydınlatma koşullarına sahip olduğu tüm koşullarda iyi olmayabilir.
    Bu durumda, uyarlamalı eşiklemeye gidiyoruz. 
    Bunda, algoritma görüntünün küçük bir bölgesi için eşiği hesaplar. 
    böylece aynı görüntünün farklı bölgeleri için farklı eşikler elde ederiz ve 
    bu bize farklı aydınlatmaya sahip görüntüler için daha iyi sonuçlar verir.
    """
    # cv2.ADAPTIVE_THRESH_MEAN_C: eşik değeri, mahalle alanının ortalamasıdır.
    thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,8)
    plt.figure()
    plt.imshow(thresh_img2, cmap="gray")
    plt.axis("off")
    plt.title("Adaptive Threshold")
    
#threshing()

def bluringImage():
    # blurring(detayı azaltır) 
    # Blur images with various low pass filters
    # https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
    img = cv2.imread("NYC.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure() 
    plt.imshow(img) 
    plt.axis("off") 
    plt.title("Orjinal Görüntü")
    
    print("1")
    """
    Averaging
    Bu, bir görüntünün normalleştirilmiş bir kutu filtresiyle sarılmasıyla yapılır. 
    Çekirdek alanı altındaki tüm piksellerin ortalamasını alır ve merkezi öğenin yerini alır.
    Bu, cv.blur () veya cv.boxFilter () işlevi tarafından yapılır. 
    Çekirdek hakkında daha fazla ayrıntı için belgelere bakın. 
    çekirdeğin genişliğini ve yüksekliğini belirtmeliyiz. 3x3 normalleştirilmiş bir kutu filtresi aşağıdaki gibi görünür:
    """
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37
    dst2 = cv2.blur(img, ksize = (3,3))
    plt.figure() 
    plt.imshow(dst2) 
    plt.axis("off") 
    plt.title("Ortalama Bulanıklaştırma")
    
    print("2")
    """Gaussian Blurring 
    
    Gaussian blurring is highly effective in removing Gaussian noise from an image.
    
    Bu yöntemde kutu filtresi yerine Gauss çekirdeği kullanılır. Bu, cv.GaussianBlur () işleviyle yapılır. Pozitif ve tek olması gereken çekirdeğin genişliğini ve yüksekliğini belirtmeliyiz. Sırasıyla sigmaX ve sigmaY X ve Y yönlerindeki standart sapmayı da belirtmeliyiz. Yalnızca sigmaX belirtilirse, sigmaY, sigmaX ile aynı şekilde alınır. Her ikisi de sıfır olarak verilirse, çekirdek boyutundan hesaplanır. Gauss bulanıklığı, bir görüntüden Gauss gürültüsünün giderilmesinde oldukça etkilidir.
    """
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
    gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
    plt.figure()
    plt.imshow(gb)
    plt.axis("off")
    plt.title("Gauss Bulanıklaştırma")
    print("3")
    
    """
    Median Blurring: This is highly effective against salt-and-pepper noise in an image.
    Burada cv.medianBlur () işlevi, çekirdek alanı altındaki tüm piksellerin medyanını alır 
    ve merkezi öğe bu medyan değerle değiştirilir. 
    Bu, bir görüntüdeki tuz ve biber gürültüsüne karşı oldukça etkilidir. 
    İlginç bir şekilde, yukarıdaki filtrelerde, merkezi eleman, görüntüdeki bir piksel değeri 
    veya yeni bir değer olabilen yeni hesaplanmış bir değerdir. 
    Ancak medyan bulanıklaştırmada, merkezi öğe her zaman görüntüdeki bazı piksel değerleriyle değiştirilir. 
    Gürültüyü etkili bir şekilde azaltır. Çekirdek boyutu pozitif bir tek tamsayı olmalıdır.
    """
    
    # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
    mb = cv2.medianBlur(img, ksize = 3)
    plt.figure()
    plt.imshow(mb)
    plt.axis("off")
    plt.title("Medyan Bulanıklaştırma")
    
    print("4")
    def gaussianNoise(image):
        
        row,col,ch= image.shape
        mean = 0
        var = 0.05
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        
        return noisy
    
    def saltPepperNoise(image):
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords] = 1
    
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords] = 0
      
        return noisy
    
    img = cv2.imread("NYC.jpg")
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original Image")
    
    print("1")
    
    gaussianNoisyImage = gaussianNoise(img)
    plt.figure()
    plt.imshow(gaussianNoisyImage)
    plt.axis("off")
    plt.title("Image with Gaussian Noise")
    
    print("2")
    spImage = saltPepperNoise(img)
    plt.figure()
    plt.imshow(spImage)
    plt.axis("off")
    plt.title("Image with Salt and Pepper Noise")
    
    print("1")
    
    # gaussian blurring
    gb = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX = 7)
    plt.figure()
    plt.imshow(gb)
    plt.axis("off")
    plt.title("Image with Gaussian Blurring")
    
    print("1")
    # median blurring
    mb = cv2.medianBlur(spImage.astype(np.float32), ksize = 3)
    plt.figure()
    plt.imshow(mb)
    plt.axis("off")
    plt.title("Image with Median Blurring")

bluringImage()