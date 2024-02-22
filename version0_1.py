# Import necessary libraries
import cv2
import threading
import queue
import numpy as np
import os
import CardDetection # Import the card detection function

# global variables
frame_queue = queue.Queue(maxsize=10)  # Queue to hold frames from the video stream


def show_opening_screen():
    # Create a black image
    opening_screen = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Set the title text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "CV Card Suite Opening Screen"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (opening_screen.shape[1] - text_size[0]) // 2
    text_y = (opening_screen.shape[0] + text_size[1]) // 2

    # Put the title text on the black image
    cv2.putText(opening_screen, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

    # Display the opening screen
    cv2.imshow("Opening Screen", opening_screen)
    cv2.waitKey(2000)  # Display the screen for 2000 milliseconds (2 seconds)
    cv2.destroyAllWindows()

""" def show_menu_screen():

    # Create a black image for the menu screen (will change later to a real menu screen)
    menu_screen = np.zeros((480, 640, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(menu_screen, "Game Selection Menu", (50, 50), font, 1, (255, 255, 255), 2)
    cv2.imshow("Menu", menu_screen)
    cv2.waitKey(2000)
    cv2.destroyAllWindows() """


def capture_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only add the frame to the queue if it's not full
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # If the queue is full, skip this frame
            pass
        cv2.waitKey(1)  # Small delay to ensure smooth video capture
    cap.release()


""" def detect_cards():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Start a thread to capture frames from the video stream
    threading.Thread(target=capture_frames, args=(cap,), daemon=True).start()
    
    cv2.namedWindow('Card Detector')
    # Main loop for processing frames
    while True:
        # Skip the loop iteration if the frame queue is empty
        if frame_queue.empty():
            continue

        # Retrieve a frame from the queue
        frame = frame_queue.get()

      
        # Show the frame
        cv2.imshow('Card Detector', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows() """
    
# Main function to execute the card detection and recognition
def main():
    show_opening_screen()
    # Start the thread to capture frames from the webcam
    capture_thread = threading.Thread(target=capture_frames) # , args=()
    capture_thread.start()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Process the frame for card detection
            detected_frame = CardDetection.detect_cards(frame)
            # Display the frame with detected cards
            cv2.imshow('Card Detector', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    
    show_menu_screen()
    detect_cards()


if __name__ == "__main__":
    main()
