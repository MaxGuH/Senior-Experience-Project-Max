# this is the main file for the Texas Holdem Poker Card Detection
#  the probability calculation still needs work but should run stand alone
#  the code is long and needs some organization/revisions but is fuctional

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
    suit_counts = {}
    ranks = []
    for card in all_cards:
        rank, suit = card.split('_')
        ranks.append(rank)
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    # Check for pair, two pair, three of a kind, and four of a kind
    pairs = []
    three_of_a_kinds = []
    for rank, count in rank_counts.items():
        if count == 2:
            pairs.append(rank)
        elif count == 3:
            three_of_a_kinds.append(rank)
        elif count == 4:
            hand_certainty['Four of a Kind'] = 100

    if pairs:
        hand_certainty['Pair'] = 100
        if len(pairs) >= 2:
            hand_certainty['Two Pair'] = 100
    if three_of_a_kinds:
        hand_certainty['Three of a Kind'] = 100

    # Check for full house
    if pairs and three_of_a_kinds:
        hand_certainty['Full House'] = 100

    # Check for flush
    for suit, count in suit_counts.items():
        if count >= 5:
            hand_certainty['Flush'] = 100
            break

    # Check for straight
    rank_order = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace']
    rank_indices = [rank_order.index(rank) for rank in ranks]
    rank_indices.sort()
    for i in range(len(rank_indices) - 4):
        if rank_indices[i + 4] - rank_indices[i] == 4:
            hand_certainty['Straight'] = 100
            break

    # Check for straight flush and royal flush
    if hand_certainty['Flush'] and hand_certainty['Straight']:
        hand_certainty['Straight Flush'] = 100
        if 'Ten' in ranks and 'Jack' in ranks and 'Queen' in ranks and 'King' in ranks and 'Ace' in ranks:
            hand_certainty['Royal Flush'] = 100

    # Calculate the highest card
    high_card_rank = max(ranks, key=lambda rank: rank_order.index(rank))
    high_card = f"High Card: {high_card_rank}"
    hand_certainty['High Card'] = high_card

    # Calculate probabilities for hands not yet certain
    # For simplicity, this example does not include detailed probability calculations for each hand.
    # You can add these calculations based on the current state of the hand and the remaining cards in the deck.

    return hand_certainty




def draw_card_selection_overlay(frame):
    global overlay_displayed
    if not overlay_displayed:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    suits = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
    values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    card_positions.clear()
    start_y = frame.shape[0] - 150  # Start 150 pixels from the bottom

    for i, suit in enumerate(suits):
        for j, value in enumerate(values):
            card_text = f"{value}{suit}"
            x, y = 10 + j * 25, start_y + i * 25
            cv2.putText(frame, card_text, (x, y), font, 0.5, (0, 0, 255), 1)  # Change color to red (BGR)
            card_positions.append(((x, y - 15, x + 20, y + 5), card_text))

def select_card(event, x, y, flags, param):
    print("select_card called")  # Add this line to check if the function is called

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
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

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

        # Display selected cards
        for i, card in enumerate(selected_cards):
            cv2.putText(frame, card, (10 + i * 75, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)

        # Label the detected cards as Flop, Turn, and River
        detected_cards.sort(key=lambda card: card[1])  # Sort by the x-coordinate
        label_cards(frame, detected_cards)

        frame_counter += 1
        
        # Calculate and display poker probabilities every 15 frames or if the number of selected cards changes
        if frame_counter % 30 == 0 or len(selected_cards) != previous_card_count:
            if selected_cards:
                print("Selected cards:", selected_cards)  # Print selected cards
                print("Filtered detected cards:", filtered_detected_cards)  # Print filter
                hand_certainty = calculate_poker_probabilities(selected_cards, filtered_detected_cards)
                display_probabilities(hand_certainty)
        previous_card_count = len(selected_cards)  # Update the count outside the if statement
        print("frame_counter", frame_counter)


        cv2.imshow('Card Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()