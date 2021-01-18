import cv2
import numpy as np

from .cvtools import resize
from .cvtools import perspective_transform
from .cvtools import getoutlines
from .cvtools import simple_erode
from .cvtools import simple_dilate
from .cvtools import brightness_contrast
from .cvtools import blank

from statistics import mean

"""
Primary Functions
"""

DARKNESS_THRESHOLD = 55
BRIGHTNESS_THRESHOLD = 80

def preprocess(img, avg_brightness):
    """
    BASIC PRE-PROCESSING TO OBTAIN A CANNY EDGE IMAGE
    """
    if(len(np.shape(img)) >= 3):
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # increase contrast between paper and background
    if(avg_brightness < DARKNESS_THRESHOLD):
      img_adj = brightness_contrast(img, 6.3, -65)
    else:
      img_adj = brightness_contrast(img, 1.68, -68)

    # calculate the ratio of the imsage to the new height (500px) so we
    # can scale the manipulated image back to the original size later
    scale = img_adj.shape[0] / 500.0

    # scale the image down to 500px in height;
    img_scaled = resize(img_adj, height=500)

    # apply gaussian blur with a 7x7 kernel
    img_gray = cv2.GaussianBlur(img_scaled, (7, 7), 0)

    # apply canny edge detection
    img_edge = cv2.Canny(img_gray, 60, 240)
    # img_edge = cv2.Canny(img_gray, 60, 250)

    # dilate the edge image to connect any small gaps
    img_edge = simple_dilate(img_edge)

    return img_adj, scale, img_scaled, img_edge


def gethull(img_edge):
    """
    1st ROUND OF OUTLINE FINDING, + CONVEX HULL
    """

    # make a copy of the edge image because the following function manipulates
    # the input
    img_prehull = img_edge.copy()

    # find outlines in the (newly copied) edge image
    outlines = getoutlines(img_prehull)

    # create a blank image for convex hull operation
    img_hull = blank(img_prehull.shape, img_prehull.dtype, "0")

    # draw convex hulls (fit polygon) for all outlines detected to 'img_contour'
    for outline in range(len(outlines)):

        hull = cv2.convexHull(outlines[outline])

        # parameters: source image, outlines (contours),
        #             contour index (-1 for all), color, thickness
        cv2.drawContours(img_hull, [hull], 0, 255, 3)

    # erode the hull image to make the outline closer to paper
    img_hull = simple_erode(img_hull)
    # show(img_hull)
    
    return img_hull


def getcorners(img_hull):
    """
    2nd ROUND OF OUTLINE FINDING, + SORTING & APPROXIMATION
    """
    
    # make a copy of the edge image because the following function manipulates
    # the input
    img_outlines = img_hull.copy()
    contours, _ = cv2.findContours(img_outlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_outlines = cv2.drawContours(img_outlines, contours, -1, (0,255,0), 3)
    # show(img_outlines)
    # find outlines in the convex hull image
    outlines = getoutlines(img_outlines)

    # sort the outlines by area from large to small, and only take the largest 4
    # outlines in order to speed up the process and not waste time
    outlines = sorted(outlines, key=cv2.contourArea, reverse=True)[:4]
    
    # loop over outlines
    for outline in outlines:

        # find the perimeter of each outline for use in approximation
        perimeter = cv2.arcLength(outline, True)

        # > approximate a rough contour for each outline found, with (hopefully)
        #   4 points (rectangular sheet of paper); [Douglas-Peuker Algorithm]
        # > FIRST OPTION is the input outline;
        # > SECOND OPTION is the accuracy of approximation (epsilon), here it
        #   is set to a percentage of the perimeter of the outline
        # > THIRD OPTION is whether to assume an outline
        #   is closed, which in this case is yes (sheet of paper)
        approx = cv2.approxPolyDP(outline, 0.02 * perimeter, True)

        # if the approximation has 4 points, then assume it is correct, and
        # assign these points to the 'corners' variable
        if len(approx) == 4:
            corners = approx
            break

    return corners