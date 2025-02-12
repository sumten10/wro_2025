import cv2 as cv
import numpy as np

def get_centroid(contour: cv.typing.MatLike):
    if len(contour) == 0:
        return
    
    M = cv.moments(contour)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    return (x, y)
    


def find_contours(image, range, ROI):

    img_segmented = image[ROI[1]:ROI[3], ROI[0]:ROI[2]]

    lower_mask = np.array(range[0])
    upper_mask = np.array(range[1])

    mask = cv.inRange(img_segmented, lower_mask, upper_mask)

    kernel = np.ones((5,5), np.uint8)

    eroded_mask = cv.erode(mask, kernel, iterations=1)
    dilated_mask = cv.dilate(eroded_mask, kernel, iterations=1)

    contours = cv.findContours(dilated_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

    return contours


def max_contour(contours, ROI):
    max_area = 0
    max_y = 0
    max_x = 0
    max_cnt = 0
    
    for c in contours:
        
        area = cv.contourArea(c)
        
        if area > 100: 
            
            approx=cv.approxPolyDP(c, 0.01*cv.arcLength(c,True),True)

            x,y,w,h=cv.boundingRect(approx)

            
            x += ROI[0] + w // 2
            y += ROI[1] + h
            
            
            if area > max_area:
                max_area = area
                max_y = y
                max_x = x
                max_cnt = c

    return [max_area, max_x, max_x, max_cnt]
