import cv2
import numpy as np
import threading
import torch
import torch.nn as nn
from torchvision import models, transforms

def determine_larger_card(card1, card2):
    card_values = {
        'Ace': 14, 'King': 13, 'Queen': 12, 'Jack': 11,
        'Ten': 10, 'Nine': 9, 'Eight': 8, 'Seven': 7,
        'Six': 6, 'Five': 5, 'Four': 4, 'Three': 3,
        'Two': 2
    } # Set a value for the back of the card
    value1 = card_values.get(card1.split('_')[0], 0)
    value2 = card_values.get(card2.split('_')[0], 0)
    return card1 if value1 > value2 else card2

class CardClassifier(nn.Module):
    def __init__(self):
        super(CardClassifier, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Change to 1 input channel
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 54)  # 54 classes for 54 cards
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)
    
# 1. Data Augmentation and Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

def preprocess_image(image, size=(256, 256)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def process_card(warped_image, model, class_names):
    preprocessed_image = preprocess_image(warped_image)
    with torch.no_grad():
        outputs = model(preprocessed_image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[predicted.item()]
        return predicted_class

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
    
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def load_model(model_path):
    model = CardClassifier()  # Create an instance of your model
    state_dict = torch.load(model_path)  # Load the state dictionary
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    model.eval()
    return model

def main():
    model_path = "CardDetection/best_model.pth"
    model = load_model(model_path)
    class_names = [
        'Ace_C', 'Two_C', 'Three_C', 'Four_C', 'Five_C',
        'Six_C', 'Seven_C', 'Eight_C', 'Nine_C', 'Ten_C',
        'Jack_C', 'Queen_C', 'King_C',
        'Ace_D', 'Two_D', 'Three_D', 'Four_D', 'Five_D',
        'Six_D', 'Seven_D', 'Eight_D', 'Nine_D', 'Ten_D',
        'Jack_D', 'Queen_D', 'King_D',
        'Ace_H', 'Two_H', 'Three_H', 'Four_H', 'Five_H',
        'Six_H', 'Seven_H', 'Eight_H', 'Nine_H', 'Ten_H',
        'Jack_H', 'Queen_H', 'King_H',
        'Ace_S', 'Two_S', 'Three_S', 'Four_S', 'Five_S',
        'Six_S', 'Seven_S', 'Eight_S', 'Nine_S', 'Ten_S',
        'Jack_S', 'Queen_S', 'King_S',
        'Joker', 'Back'
    ]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow('War Card Detector', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('War Card Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    max_cards = 2
    points_card1 = 26
    points_card2 = 26
    new_cards_detected = True 
    points = 1
    tie_flag = False

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_cards = []

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if len(approx) == 4 and area > 4000 and 0.7 < aspect_ratio < 1.3:
                warped_image = four_point_transform(frame, approx.reshape(4, 2))
                predicted_class = process_card(warped_image, model, class_names)
                detected_cards.append((predicted_class, (x, y, w, h)))
                
                #cv2.polylines(frame, [approx], True, (0, 255, 0), 2)

                if len(detected_cards) >= max_cards:
                    break

        # If two cards are detected, determine the larger card
        if len(detected_cards) == max_cards:
            detected_cards.sort(key=lambda x: x[1][0])
            card1, (x1, y1, w1, h1) = detected_cards[0]
            card2, (x2, y2, w2, h2) = detected_cards[1]

            scard1 = card1.split('_')[0]
            scard2 = card2.split('_')[0]

            if scard1 == scard2:
                # Draw a blue border around both cards if they are equal
                cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 3)
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'war', (100, 100), font, 5, (0, 0, 0), 5, cv2.LINE_AA)
    
                cv2.imshow('Frame', frame)
                cv2.waitKey(2000)  
                cv2.destroyWindow('Frame')

                cv2.putText(frame, '', (50, 50), font, 2, (0, 0, 0), 5, cv2.LINE_AA)

                # cv2.putText(frame, 'Remove the cards from the frame', (50, 50), font, 2, (0, 0, 0), 5, cv2.LINE_AA)

                # cv2.imshow('Frame', frame)
                # cv2.waitKey(2000)
                # cv2.destroyWindow('Frame')

                points += 2
                tie_flag = True

            else:
                larger_card = determine_larger_card(card1, card2)

                if new_cards_detected:
                    if larger_card == card1:
                        points_card1 += points
                        points_card2 -= points
                    elif larger_card == card2:
                        points_card1 -= points
                        points_card2 += points
                    new_cards_detected = False
                    if tie_flag:
                        points = 1
                        tie_flag = False
                    

                # Highlight the larger card (draw a border)
                for card, (x, y, w, h) in detected_cards:
                    if card == larger_card:
                        # Draw a green border around the larger card
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    else:
                        # Draw a red border around the smaller card
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        elif len(detected_cards) == 0:
            new_cards_detected = True
        cv2.putText(frame, f"Card 1 Points: {points_card1}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Card 2 Points: {points_card2}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('War Card Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
