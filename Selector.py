import war
from CardDetection import Crazy4s
import tkinter as tk
from tkinter import PhotoImage

crazy4_option = None
war_option = None
fish_option = None
second_window = None

def on_war_click():
    war.main()
    war_option.destroy()

def on_fish_click():
    fish_option.destroy()

def on_crazy4_click():
    crazy4_option.destroy()
    Crazy4s.main()

def close_window():
    root.destroy()
    open_second_window()

def open_second_window():
    # Create a new top-level window
    global second_window
    second_window = tk.Toplevel()
    second_window.title("Second Window")

    # Configure the background color of the second window
    second_window.configure(bg='#00512c')

    # Set the geometry of the second window
    screen_width = second_window.winfo_screenwidth()
    screen_height = second_window.winfo_screenheight()
    second_window.geometry(f"{screen_width}x{screen_height}")

    # Create buttons in the new window
    button1 = tk.Button(second_window, text="War", command=option1_click, bg = 'red', fg='white', width=20, height=5)
    button1.place(x=screen_width - 150, y=50)
    button1.pack(pady=10)

    button2 = tk.Button(second_window, text="Poker", command=option2_click, bg = 'black', fg='white',width=20, height=5)
    button2.pack(pady=10)

    button3 = tk.Button(second_window, text="Crazy 4's", command=option3_click, bg = 'red', fg='white',width=20, height=5) 
    button3.pack(pady=10)

    button4 = tk.Button(second_window, text="Black Jack", command=option4_click, bg = 'black', fg='white',width=20, height=5)
    button4.pack(pady=10)

    button5 = tk.Button(second_window, text="Solitare", command=option5_click, bg = 'red', fg='white', width=20, height=5)
    button5.pack(pady=10)

def option1_click():
    second_window.destroy()
    global war_option
    war_option = tk.Toplevel()
    war_option.title("Rules of War")

    screen_width = war_option.winfo_screenwidth()
    screen_height = war_option.winfo_screenheight()
    war_option.geometry(f"{screen_width}x{screen_height}")

    text = "The Game of War \n Take the deck and split it evenly between the players \n Make sure to do this without looking at the face of the cards \n After dealing the cards, both players play the card on top of their card pile \n"
    text = text + "After both players play their top card, compare the two cards \n Which ever card's rank is higher wins \n If the ranks are the same, then you go to war \n"
    war_label = tk.Label(war_option, text=text)

    # Create a button
    war_button = tk.Button(war_option, text="Play War", command=on_war_click)

    war_label.pack(padx=20, pady=20)
    war_button.pack(side=tk.BOTTOM, padx=20, pady=20)

def option2_click():
    second_window.destroy()
    global fish_option
    fish_option = tk.Toplevel()
    fish_option.title("Rules of Poker")

    screen_width = fish_option.winfo_screenwidth()
    screen_height = fish_option.winfo_screenheight()
    fish_option.geometry(f"{screen_width}x{screen_height}")

    text = "Joe has ligma"
    fish_label = tk.Label(fish_option, text=text)

    fish_button = tk.Button(fish_option, text="Play Fish", command=on_fish_click)

    fish_label.pack(padx=20, pady=20)
    fish_button.pack(side=tk.BOTTOM, padx=20, pady=20)


def option3_click():
    second_window.destroy()
    global crazy4_option
    crazy4_option = tk.Toplevel()
    crazy4_option.title("Rules of War")

    screen_width = crazy4_option.winfo_screenwidth()
    screen_height = crazy4_option.winfo_screenheight()
    crazy4_option.geometry(f"{screen_width}x{screen_height}")

    text = "Crazy 4's \n The objective of the game is to match similar ranks (The face of the card) and suit \n If a card has a match to the card in the play card pile, you may play it \n"
    text = text + "4's are considered wild cards. You may play these at any time"
    fours_label = tk.Label(crazy4_option, text=text)

    # Create a button
    fours_button = tk.Button(crazy4_option, text="Play Crazy 4's", command=on_crazy4_click)

    fours_label.pack(padx=20, pady=20)
    fours_button.pack(side=tk.BOTTOM, padx=20, pady=20)

def option4_click():
    print("Black Jack selected")

def option5_click():
    print("Solitare selected")

root = tk.Tk()
root.title("Title Screen Logo")

image_path = "Stuff/Logo.png"
background_image = PhotoImage(file=image_path)

image_width = background_image.width()
image_height = background_image.height()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

canvas.create_image(0,0, anchor=tk.NW, image=background_image)

root.after(2000, close_window)

root.mainloop()
