# this is the main file for the Texas Holdem Poker Card Detection
#  the probability calculation still needs work but should run stand alone
#  the code is long and needs some organization/revisions but is functional

import torch
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import threading

selected_cards = []
card_positions = []
overlay_displayed = True
detected_cards = []

def filter_cards(cards):
    return [card for card in cards if card not in ('Back', 'Joker')]

def label_cards(frame, detected_cards):
    for i, (card, x, y, w, h) in enumerate(detected_cards):
        label = None
        if i < 3:
            label = "Flop"
        elif i == 3:
            label = "Turn"
        elif i == 4:
            label = "River"

        if label is not None:
            # Adjust the y-coordinate to display the label at the bottom of the card
            cv2.putText(frame, label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


def display_probabilities(hand_certainty):
    print("display probabilities called")
    prob_img = np.zeros((400, 300, 3), dtype=np.uint8)  # Increase the height to accommodate more text
    for i, (hand, certainty) in enumerate(hand_certainty.items()):
        if hand == 'High Card':
            certainty_text = f"{certainty}"
        else:
            certainty_text = f"{hand}: {certainty}%"
        cv2.putText(prob_img, certainty_text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Probabilities', prob_img)
    cv2.waitKey(1)

def calculate_poker_probabilities(selected_cards, filtered_detected_cards):
    hand_certainty = {
        'High Card': 0,
        'Pair': 0,
        'Two Pair': 0,
        'Three of a Kind': 0,
        'Straight': 0,
        'Flush': 0,
        'Full House': 0,
        'Four of a Kind': 0,
        'Straight Flush': 0,
        'Royal Flush': 0
    }

    all_cards = selected_cards + filtered_detected_cards
    all_cards = [card for card in all_cards if card not in ['Back', 'Joker'] and '_' in card]

    if not all_cards:
        return hand_certainty

    rank_counts = {}
    for card in all_cards:
        rank, _ = card.split('_')
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    num_community_cards = len(all_cards) - 2  # Subtract 2 for your hand cards

    # Update probabilities based on the current hand
    pair_count = 0
    three_count = 0
    four_count = 0

    for rank, count in rank_counts.items():
        if count == 2:
            pair_count += 1
            hand_certainty['Pair'] = 100
        elif count == 3:
            three_count += 1
            hand_certainty['Three of a Kind'] = 100
        elif count == 4:
            four_count += 1
            hand_certainty['Four of a Kind'] = 100

    if pair_count == 2:
        hand_certainty['Pair'] = 100
        hand_certainty['Two Pair'] = 100

    if pair_count == 2 and three_count == 1:
        hand_certainty['Two Pair'] = 100
        hand_certainty['Three of a Kind'] = 100
        hand_certainty['Full House'] = 100

    # Update probabilities based on the stage of the game
    if pair_count == 0:
        if num_community_cards == 0:  # Pre-flop
            hand_certainty['Pair'] = max(hand_certainty['Pair'], 69.62)
            hand_certainty['Two Pair'] = max(hand_certainty['Two Pair'], 23.41)
            hand_certainty['Three of a Kind'] = max(hand_certainty['Three of a Kind'], 2.11)
            hand_certainty['Four of a Kind'] = max(hand_certainty['Four of a Kind'], 0.168)
        elif num_community_cards == 3:  # Flop
            hand_certainty['Pair'] = max(hand_certainty['Pair'], 20.43)
            hand_certainty['Two Pair'] = max(hand_certainty['Two Pair'], 7.02)
            hand_certainty['Three of a Kind'] = max(hand_certainty['Three of a Kind'], 0.88)
            hand_certainty['Four of a Kind'] = max(hand_certainty['Four of a Kind'], 0)
        elif num_community_cards == 4:  # Turn
            hand_certainty['Pair'] = max(hand_certainty['Pair'], 39.13)
            hand_certainty['Two Pair'] = max(hand_certainty['Two Pair'], 0)
            hand_certainty['Three of a Kind'] = max(hand_certainty['Three of a Kind'], 0)
            hand_certainty['Four of a Kind'] = max(hand_certainty['Four of a Kind'], 0)
    if pair_count == 1:
        if num_community_cards == 0:  # Pre-flop
            hand_certainty['Pair'] = 100
            hand_certainty['Two Pair'] = max(hand_certainty['Two Pair'], 55.32)
            hand_certainty['Three of a Kind'] = max(hand_certainty['Three of a Kind'], 39.20 )
            hand_certainty['Four of a Kind'] = max(hand_certainty['Four of a Kind'], 0.354)

    return hand_certainty


def draw_card_selection_overlay(frame):
    global overlay_displayed
    if not overlay_displayed:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    suits = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
    values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    card_positions.clear()
    start_y = frame.shape[0] - 200  # Start 150 pixels from the bottom

    for i, suit in enumerate(suits):
        for j, value in enumerate(values):
            card_text = f"{value}{suit}"
            x, y = 10 + j * 40, start_y + i * 30
            cv2.putText(frame, card_text, (x, y), font, 0.5, (0, 0, 255), 1)  # Change color to red (BGR)
            card_positions.append(((x, y - 15, x + 20, y + 5), card_text))

def select_card(event, x, y, flags, param):
    # print("select_card called")  # Add this line to check if the function is called

    global overlay_displayed
    if event == cv2.EVENT_LBUTTONDOWN:
        for position, card_text in card_positions:
            x1, y1, x2, y2 = position
            if x1 < x < x2 and y1 < y < y2:
                rank, suit = card_text[:-1], card_text[-1]
                rank_full_name = {
                    'A': 'Ace', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five',
                    '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine', '10': 'Ten',
                    'J': 'Jack', 'Q': 'Queen', 'K': 'King'
                }[rank]
                selected_card = f"{rank_full_name}_{suit}"
                if selected_card not in selected_cards:
                    selected_cards.append(selected_card)
                if len(selected_cards) >= 2:
                    overlay_displayed = False  # Hide overlay after selecting two cards
                break


def process_card(warped_image, model, class_names, x, y, frame):
    preprocessed_image = preprocess_image(warped_image)
    with torch.no_grad():
        outputs = model(preprocessed_image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[predicted.item()]
        print(f"Predicted class: {predicted_class}")
    cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



# 3. Model Definition
class CardClassifier(nn.Module):
    def __init__(self):
        super(CardClassifier, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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
    model_path = "CardDetection\\best_model.pth"
    model = load_model(model_path)
    detected_cards = []

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    
    cv2.namedWindow('Card Detector', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Card Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Card Detector', select_card)

    max_cards = 5
    frame_counter = 0
    previous_card_count = 0
    previous_detected_cards_count = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_cards.clear()
        draw_card_selection_overlay(frame)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_threads = []
        for contour in contours:
    # Process each contour...
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if len(approx) == 4 and area > 4000 and 0.7 < aspect_ratio < 1.3:
                warped_image = four_point_transform(frame, approx.reshape(4, 2))
    # Process the card to get its class
                preprocessed_image = preprocess_image(warped_image)
                with torch.no_grad():
                    outputs = model(preprocessed_image)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_class = class_names[predicted.item()]

                # Draw a green bounding box around the detected card
                cv2.polylines(frame, [approx], True, (0, 255, 0), 2)
                cv2.putText(frame, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if len(detected_cards) < 5:  # Only detect the first five cards
                    detected_cards.append((predicted_class, x, y, w, h))  # Store the card # Store the card class and position
        filtered_detected_cards = filter_cards([card for card, _, _, _, _ in detected_cards])

        # # Display selected cards
        # for i, card in enumerate(selected_cards):
        #     cv2.putText(frame, card, (10 + i * 75, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
        # Set the spacing between cards and the vertical offset for the labels
        card_spacing = 150
        label_offset = 25

        
        # Loop through the selected cards and draw the elements
        for i, card in enumerate(selected_cards):
            # Display "Your Hand" text above the cards
            cv2.putText(frame, "Your Hand", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)

            # Calculate the horizontal position for this card based on the index and spacing
            card_pos_x = 10 + i * card_spacing

            # Display the card name below the rectangle icon
            cv2.putText(frame, card, (10 + i * 80, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0), 1)



        # Label the detected cards as Flop, Turn, and River
        detected_cards.sort(key=lambda card: card[1])  # Sort by the x-coordinate
        label_cards(frame, detected_cards)

        frame_counter += 1
        
        # Calculate and display poker probabilities every 30 frames or if the number of selected cards changes
        if frame_counter % 30 == 0 or len(selected_cards) != previous_card_count:
            if selected_cards:
                print("Selected cards:", selected_cards)  # Print selected cards
                print("Filtered detected cards:", filtered_detected_cards)  # Print filter
                hand_certainty = calculate_poker_probabilities(selected_cards, filtered_detected_cards)
                display_probabilities(hand_certainty)
        previous_card_count = len(selected_cards)  # Update the count outside the if statement
        # print("frame_counter", frame_counter)


        cv2.imshow('Card Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()