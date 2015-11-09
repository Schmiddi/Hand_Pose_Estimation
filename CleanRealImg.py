'''
Created on 06.11.2015
'''

import cv2
import numpy as np
import copy
import math

MIN_AREA_SIZE = 2000
THRESH_DEPTH = 20000
TARGET_MEDIAN = 26000

def contourFilter(contour_bound):
    _, _, w, h = contour_bound[1]
    check = w * h > MIN_AREA_SIZE \
                and float(w) / h <= 1.3
    return check

def contourFilter2(contour_bound):
    _, _, w, h = contour_bound[1]
    check = w * h > MIN_AREA_SIZE 
                # and float(w) / h <= 1.3
    return check
def removeBackground(img):
    # Invert color:
    img = 65535 - img
    img[img < THRESH_DEPTH] = 0
    # Filter the noise
#     newImg = copy.copy(img)
#     cv2.filter2D(img, newImg, cv2.CV_16U,)
    # Media filter the img
    # img = cv2.medianBlur(img, 5)
    
    newImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    newImg[img > THRESH_DEPTH] = 255
    # crop the bottom:
    newImg[-20:, :] = 0
    bw = copy.copy(newImg)
#     cv2.imshow('img', newImg)
#     cv2.waitKey(0)
    contours, _ = cv2.findContours(newImg, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    contour_bounds = [(c, cv2.boundingRect(c)) for c in contours]
    contour_bounds = filter(contourFilter, contour_bounds)
#     print len(contour_bounds)
    
    # If more than one large contour has been found, use the one closer to the center of the image
    imax = 0
    center = np.array([img.shape[0]/2,img.shape[1]/2])
    
    if len(contour_bounds) > 1:
        lenCB = len(contour_bounds)
        # Choose the largest:
        distance = np.zeros(lenCB)
        for i in xrange(lenCB):
            # Height is more important, therefore we square it
            x, y, w, h = contour_bounds[i][1]
            contCenter = np.array([y+h/2, x+w/2])
            distance[i] = np.linalg.norm(center - contCenter)
        
        imax = np.argmin(distance)
        
    # Remove everything outside of the contour rectangle
    x, y, w, h = contour_bounds[imax][1]
    pt1 = (x, y)
    pt2 = (x + w, y + h - 1)
    mask = np.zeros_like(newImg, dtype=np.uint8)
    cv2.rectangle(mask, pt1, pt2, (255, 255, 255), cv2.cv.CV_FILLED)
    img[mask == 0] = 0
     
#     cv2.imshow('mask', mask)
#     cv2.waitKey(0)
    # Remove everything from the bottom until the width shrinks
    for i in xrange(0 + 1, h):
        # print "Mean: ", np.mean(bw[y+h-i, x:x+w])
        # print newImg[y+h-i, x:x+w]
        if np.mean(bw[y + h - i, x:x + w]) > 150:
            img[y + h - i, x:x + w] = 0
        else:
            for j in xrange(10):
                img[y + h - i - j, x:x + w] = 0
            break
    
    
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
    
    # Remove only the outside noise using the mask
    newImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    newImg[img > THRESH_DEPTH] = 255
    newImgBlur = cv2.medianBlur(newImg, 3)
#     cv2.imshow('img', newImg)
#     cv2.imshow('img', newImgBlur)
#     cv2.waitKey(0)
    img[newImgBlur == 0] = 0
    
    # Crop the image, having the hand in the center
    contours, _ = cv2.findContours(newImgBlur, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
     
    contour_bounds = [(c, cv2.boundingRect(c)) for c in contours]
    contour_bounds = filter(contourFilter2, contour_bounds)
    
#     print len(contour_bounds)
    
    if len(contour_bounds) > 1:
        exit(0)
        
    x, y, w, h = contour_bounds[0][1]
#     cv2.rectangle(img, (x, y), (x + w, y + h), (65535, 65535))
#     cv2.imshow('rec', img)
#     cv2.waitKey(0)
    # print x,y,w,h
    img = img[y + (h / 2) - 80:y + (h / 2) + 80, x + (w / 2) - 80:x + (w / 2) + 80]     

    # Check if the img is 160x160
    rowOff = 0 if img.shape[0] == 160 else math.ceil((160 - img.shape[0]) / 2.0)
    colOff = 0 if img.shape[1] == 160 else math.ceil((160 - img.shape[1]) / 2.0)
    
    # Adjust it if it isn't, if at least one dimension is wrong
    if rowOff != 0 or colOff != 0:
        tmp = np.zeros((160,160),dtype=np.uint16)
        tmp[rowOff:img.shape[0]+rowOff, colOff:img.shape[1]+colOff] = img
        img = tmp  
    
    # ------------ Remove the wire ----------------
    # Before every part the must be at least 2 bg pixel, then remove every part which is less 
    # than 5 px long, therewith the wire can be removed
    
    
    newImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    newImg[img > THRESH_DEPTH] = 255
    
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
    
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
    start = None
    minRow = int((80 - h / 2) * 1.5)
#     print minRow
    for row in xrange(minRow, newImg.shape[0] - 3):
        for col in xrange(newImg.shape[1] / 2, newImg.shape[1] - 2):
            if start is None and newImg[row, col] == 255 and np.mean(newImg[row, col - 2:col]) == 0:
                start = col
            elif start is not None and np.mean(newImg[row, col:col + 2]) == 0:
                if col - start < 5:
                    newImg[row, start:col] = 0
                start = None
    
    for col in xrange(newImg.shape[1] / 2, newImg.shape[1]):
        start = None
        for row in xrange(minRow, newImg.shape[0] - 2):
            if start is None and newImg[row, col] == 255 and np.mean(newImg[row - 2:row, col]) == 0:
                start = row
            elif start is not None and np.mean(newImg[row:row + 2, col]) == 0:
                if row - start < 6:
                    newImg[start:row, col] = 0
                start = None
    
    for row in xrange(minRow, newImg.shape[0] - 3):
        for col in xrange(newImg.shape[1] / 2, newImg.shape[1] - 2):
            if start is None and newImg[row, col] == 255 and np.mean(newImg[row, col - 2:col]) == 0:
                start = col
            elif start is not None and np.mean(newImg[row, col:col + 2]) == 0:
                if col - start < 5:
                    newImg[row, start:col] = 0
                start = None
                
    for col in xrange(newImg.shape[1] / 2, newImg.shape[1]):
        start = None
        for row in xrange(minRow, newImg.shape[0] - 2):
            if start is None and newImg[row, col] == 255 and np.mean(newImg[row - 2:row, col]) == 0:
                start = row
            elif start is not None and np.mean(newImg[row:row + 2, col]) == 0:
                if row - start < 5:
                    newImg[start:row, col] = 0
                start = None
    img[newImg == 0] = 0
    newImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    newImg[img > THRESH_DEPTH] = 255
    # Remove all small areas
    contours, _ = cv2.findContours(newImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
    contour_bounds = [(c, cv2.boundingRect(c)) for c in contours]
    for contour in contour_bounds:
        x, y, w, h = contour[1]
        if w * h < MIN_AREA_SIZE and w * h > 20:
            img[y:y + h, x:x + w] = 0
    
    #### ------------ Adjust the depth of the image -------------
    # Get the media of all non 0 pixels
    handMedian = np.median(img[img!=0])
    diff = TARGET_MEDIAN - handMedian
    img[img!=0] += diff
    
    return img

if __name__ == '__main__':
    # Create the list of all indices
    # 1 - 10, A-Y
    # Except: 3, 7, J (74), R (82), T (84), W (87)
   
    indices = [str(i) for i in xrange(1, 11) if i not in [3, 7]] + [chr(i) for i in xrange(65, 90) if i not in [74, 82, 84, 87]]
    
    for j in xrange(1,31): #[21]:#
        for i in indices: #['1','2','4','5','6','8']:# ['6']:#
            filename = 'subject-' + str(j) + '_' + i + '.skdepth.png'
            print filename 
            img = cv2.imread('RealImages/' + filename, -1)  # , cv2.CV_LOAD_IMAGE_GRAYSCALE)
            img = removeBackground(img)
            cv2.imwrite('ProcessedRealImages/16bit/' + filename, img)

            img[img == 0] = 20000
            img = 65535 - img
            img /= ((65535-20000)/750.0)
            cv2.imwrite('ProcessedRealImages/scaled/' + filename, img)
    

'''
    Reproduce syntetic images:
        - 160x160 - 80,80 = center of the palm [x]
        - inverted colors (black is background) [x]
        - Crop such that the hand is in the middle [x]
        - Remove the wire [x]
        - Adjust the depth of the hand - every hand should be the same [x]
        - Generate a completly black image - can can [x]
        - maybe scale hand [x]
'''
