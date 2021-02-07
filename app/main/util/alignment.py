from typing import Tuple, Union
import math
import numpy as np
import cv2
import pytesseract
import re

import app.lib.pyimgscan.tools as tools

#pytesseract.pytesseract.tesseract_cmd = '/usr/share/tesseract-ocr'

def rotate(image=np.ndarray, angle=float, background=Union[int, Tuple[int, int, int]]):
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def findAngle(image, documentType, center = None, scale = 1.0):
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    documentconfig = ''
    
    # Set the config of Tesseract depending on the type of document
    if documentType == 'document':
        documentconfig = ''
    else:
        documentconfig = '--dpi 300 --psm 0 -c min_characters_to_try=1'

    angle = 360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image, config=documentconfig)).group(0))

    return angle


#USED
def deskew_img(image, documentType):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur the image
    #grayscale = cv2.GaussianBlur(grayscale, (9, 9), 0)
    
    # Using deskew (old solution for identifying the angle)
    # angle = determine_skew(grayscale)
    
    # Tresseract solution for finding the angle of the image
    angle = findAngle(image, documentType)
    rotated = rotate(grayscale, angle, (0, 0, 0))
    deskewed_img = rotated.astype(np.uint8)
    padded_img = np.pad(deskewed_img, pad_width=30, mode='constant', constant_values=0)
    avg_brightness = np.mean(grayscale)

    return padded_img, avg_brightness

#USED
def pyscanimg(image, avg_brightness):
    """
    Main Proccess of the Program
    """
    img_adj, scale, _, img_edge = tools.preprocess(image, avg_brightness)
    img_hull = tools.gethull(img_edge)
    corners = tools.getcorners(img_hull)
    corners = corners.reshape(4, 2) * scale
    img_corrected = tools.perspective_transform(img_adj, corners)
    return img_corrected

#USED
def remove_shadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    return cv2.merge(result_planes)

#USED
def sharpen(img_grey):
    kernel = np.array([[0, -1, 0], \
                      [-1, 5,-1], \
                      [0, -1, 0]])
    return cv2.filter2D(img_grey, -1, kernel)

def align(img, documentType="", color="bw"):
    color = color.lower()
    documentType = documentType.lower()
    image_rotated, avg_brightness = deskew_img(img, documentType)
    image_corrected = pyscanimg(image_rotated, avg_brightness)
    image_filtered = remove_shadow(image_corrected)
    image_sharpened = sharpen(image_filtered)
    if color == "bw":
        (_, blackAndWhiteImage) = cv2.threshold(image_sharpened, 240, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return blackAndWhiteImage
    elif color == "grey" or color == "gray":
        return image_sharpened
    else:
        return None
