import cv2
import numpy as np
import version0

#TODO: Add original source the code came from.

def detect_cards(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter for card-sized contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if 2000 < area < 60000:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # If the polygon has four sides, we can assume it's a card
            if len(approx) == 4:
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 4)
    return frame
