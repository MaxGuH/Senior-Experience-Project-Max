import cv2
import numpy as np

captured_frame = None

def get_card_rank():
    # Code to find the card Rank
    rank = 0

    if rank == "Jack":
        rank  = 11
    elif rank == "Queen":
        rank = 12
    elif rank == "King":
        rank = 13
    elif rank == "Ace":
        rank = 14

    return 0

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
                # 2 variables player_left and player_right for a counter
                # 2 variables card_left and card_right for card detection
                card_rank = get_card_rank()
                color = 0

                if card_rank > 1: # The card with the higher rank will be highlighted green
                    color = (0, 255, 0) # Green
                elif 1 < card_rank: # The lesser of the card ranks will be highlighted red
                    color = (0, 0, 255) # Red
                else: # If the cards are of equal rank or an error happens, the cards will be highlighted blue
                    color = (255, 0, 0) # Blue
                cv2.drawContours(frame, [approx], 0, color, 4)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                font_color = (255, 255, 255)

                # Evenually change the number 26 to the counters above to keep track of cards in player hand
                cv2.putText(frame, f"26", (10, 30), font, font_scale, font_color, font_thickness)
                cv2.putText(frame, f"26", (frame.shape[1] - 100, 30), font, font_scale, font_color, font_thickness)

    return frame

def open_camera():
    # Open a camera capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        _, frame = cap.read()

        # Call the function to detect and highlight cards
        highlighted_frame = detect_cards(frame)

        # Display the frame
        cv2.imshow('Card Detection', highlighted_frame)

        if captured_frame is not None:
            cv2.imshow("Captured Frame", captured_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

open_camera()