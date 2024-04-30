import tkinter as tk
from tkinter import PhotoImage, Toplevel, font as tkfont
import threading
import war
from CardDetection import Crazy4s, TexasHoldem, Blackjack

def on_war_click():
    war.main()
    print("Starting War...")

def on_fish_click():
    print("Starting Poker...")

def on_crazy4_click():
    loading_window = Toplevel()
    loading_window.title("Loading Crazy 4's...")
    loading_window.attributes('-fullscreen', True)
    loading_image = PhotoImage(file="Stuff/Cr4menu.png")
    loading_label = tk.Label(loading_window, image=loading_image)
    loading_label.image = loading_image
    loading_label.pack(expand=True, fill=tk.BOTH)

    def start_game():
        Crazy4s.main()
        print("Starting Crazy 4's...")
        loading_window.destroy()

    thread = threading.Thread(target=start_game)
    thread.daemon = True
    thread.start()
    loading_window.after(8000, loading_window.destroy)

def on_poker_click():
    loading_window = Toplevel()
    loading_window.title("Loading Crazy Texas Hold'em...")
    loading_window.attributes('-fullscreen', True)
    loading_image = PhotoImage(file="Stuff/pokermenu.png")
    loading_label = tk.Label(loading_window, image=loading_image)
    loading_label.image = loading_image
    loading_label.pack(expand=True, fill=tk.BOTH)

    def start_game():
        TexasHoldem.main()
        print("Starting Texas Hold'em...")
        loading_window.destroy()

    thread = threading.Thread(target=start_game)
    thread.daemon = True
    thread.start()
    loading_window.after(8000, loading_window.destroy)

def on_blackjack_click():
    loading_window = Toplevel()
    loading_window.title("Loading BlackJack...")
    loading_window.attributes('-fullscreen', True)
    loading_image = PhotoImage(file="Stuff/pokermenu.png")
    loading_label = tk.Label(loading_window, image=loading_image)
    loading_label.image = loading_image
    loading_label.pack(expand=True, fill=tk.BOTH)

    def start_game():
        Blackjack.main()
        print("Starting BlackJack...")
        loading_window.destroy()

    thread = threading.Thread(target=start_game)
    thread.daemon = True
    thread.start()
    loading_window.after(8000, loading_window.destroy)

def option1_click():
    global war_option
    war_option = Toplevel()
    war_option.title("Rules of War")
    screen_width = war_option.winfo_screenwidth()
    screen_height = war_option.winfo_screenheight()
    war_option.geometry(f"{screen_width}x{screen_height}")
    large_font = ('Verdana', 16)
    text = """The Game of War
Take the deck and split it evenly between the players
Make sure to do this without looking at the face of the cards
After dealing the cards, both players play the card on top of their card pile
After both players play their top card, compare the two cards
Whichever card's rank is higher wins
If the ranks are the same, then you go to war"""
    war_label = tk.Label(war_option, text=text, font=large_font)
    war_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
    war_button = tk.Button(war_option, text="Play War", command=on_war_click, font=large_font)
    war_button.pack(side=tk.TOP, padx=20, pady=10)
    back_button = tk.Button(war_option, text="Back", command=back_to_menu, font=large_font)
    back_button.pack(side=tk.TOP, padx=20, pady=10)

def back_to_menu():
    global war_option, crazy4_option, poker_option
    if 'war_option' in globals() and war_option:
        war_option.destroy()
    if 'crazy4_option' in globals() and crazy4_option:
        crazy4_option.destroy()
    if 'poker_option' in globals() and poker_option:
        poker_option.destroy()

def option2_click():
    global poker_option
    poker_option = Toplevel()
    poker_option.title("Rules of Poker")
    screen_width = poker_option.winfo_screenwidth()
    screen_height = poker_option.winfo_screenheight()
    poker_option.geometry(f"{screen_width}x{screen_height}")
    large_font = ('Verdana', 16)

    text = "There are lots of rules in poker."
    poker_label = tk.Label(poker_option, text=text, font=large_font)
    poker_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
    poker_button = tk.Button(poker_option, text="Play Texas Hold'em", command=on_poker_click, font=large_font)
    poker_button.pack(side=tk.TOP, padx=20, pady=10)
    back_button = tk.Button(poker_option, text="Back", command=back_to_menu, font=large_font)
    back_button.pack(side=tk.TOP, padx=20, pady=10)

def option3_click():
    global crazy4_option
    crazy4_option = Toplevel()
    crazy4_option.title("Rules of Crazy 4's")
    screen_width = crazy4_option.winfo_screenwidth()
    screen_height = crazy4_option.winfo_screenheight()
    crazy4_option.geometry(f"{screen_width}x{screen_height}")
    large_font = ('Verdana', 16)
    text = "Crazy 4's \nThe objective of the game is to match similar ranks and suit.\n" \
           "If a card has a match to the card in the play card pile, you may play it.\n" \
           "4's are considered wild cards. You may play these at any time."
    fours_label = tk.Label(crazy4_option, text=text, font=large_font)
    fours_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
    fours_button = tk.Button(crazy4_option, text="Play Crazy 4's", command=on_crazy4_click, font=large_font)
    fours_button.pack(side=tk.TOP, padx=20, pady=10)
    back_button = tk.Button(crazy4_option, text="Back", command=back_to_menu, font=large_font)
    back_button.pack(side=tk.TOP, padx=20, pady=10)

def option4_click():
    global crazy4_option
    crazy4_option = Toplevel()
    crazy4_option.title("Rules of Black Jack")
    screen_width = crazy4_option.winfo_screenwidth()
    screen_height = crazy4_option.winfo_screenheight()
    crazy4_option.geometry(f"{screen_width}x{screen_height}")
    large_font = ('Verdana', 16)
    text = "Blackjack.\nThe goal is to beat the dealer by having a hand total that does not exceed 21.\n" \
           "Players are dealt two cards and can choose to 'Hit' to take additional cards or 'Stand' to maintain their current total.\n" \
           "Face cards are worth 10 points, Aces can be worth 1 or 11, and all other cards are valued by their number."
    fours_label = tk.Label(crazy4_option, text=text, font=large_font)
    fours_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
    fours_button = tk.Button(crazy4_option, text="Play Blackjack", command=on_blackjack_click, font=large_font)
    fours_button.pack(side=tk.TOP, padx=20, pady=10)
    back_button = tk.Button(crazy4_option, text="Back", command=back_to_menu, font=large_font)
    back_button.pack(side=tk.TOP, padx=20, pady=10)

def option5_click():
    print("Solitaire selected")


import tkinter as tk
from tkinter import PhotoImage, Toplevel, font as tkfont

def open_main_window():
    global second_window
    if not 'second_window' in globals() or not second_window.winfo_exists():
        # Create second_window if it does not exist or has been closed
        second_window = Toplevel(root)
        second_window.title("Game Selection")
        second_window.configure(bg='#00512c')
        second_window.attributes('-fullscreen', True)
        setup_main_window(second_window)


from tkinter import PhotoImage, Toplevel, font as tkfont, Label
from PIL import Image, ImageTk
import tkinter as tk

def setup_main_window(window):
    # Load the original image with PIL
    original = Image.open("Stuff/projectLogo.png")
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the new size: we want the logo height to be about 25% of screen height
    new_height = screen_height // 4
    new_width = int(original.width * new_height / original.height)  # Maintain aspect ratio

    # Resize the image using PIL with the correct attribute for high-quality downsampling
    resized = original.resize((new_width, new_height), Image.LANCZOS)  # Use Image.LANCZOS for high-quality downsampling

    # Convert the PIL image to PhotoImage
    root.images['project_logo'] = ImageTk.PhotoImage(resized)
    project_logo_label = Label(window, image=root.images['project_logo'], bg='#00512c')
    project_logo_label.pack(side=tk.TOP, pady=20)

    button_font = tkfont.Font(family='Helvetica', size=24, weight='bold')
    buttons = [
        ("War", option1_click, 'red'),
        ("Texas Hold'em Poker", option2_click, 'black'),
        ("Crazy 4's", option3_click, 'red'),
        ("Black Jack", option4_click, 'black'),
        ("Solitaire", option5_click, 'red')
    ]
    for text, command, color in buttons:
        button = tk.Button(window, text=text, command=command, bg=color, fg='white', font=button_font)
        button.pack(pady=10, padx=100, fill=tk.BOTH, expand=True)

    exit_button = tk.Button(window, text="Exit", command=window.destroy, bg='black', fg='white', font=button_font)
    exit_button.pack(side=tk.BOTTOM, anchor=tk.W, padx=10, pady=10)

def close_window():
    root.withdraw()  # Hide the root window
    open_main_window()
def close_all_windows():
    root.quit()  # This will handle the destruction of all widgets and quit the mainloop


def show_project_logo():
    # Display the initial logo
    root.images['logo'] = PhotoImage(file="Stuff/Logo.png")
    logo_label = tk.Label(root, image=root.images['logo'], bg='black')
    logo_label.place(x=0, y=0, relwidth=1, relheight=1)
    root.after(2000, switch_logo)

def switch_logo():
    # Change to another logo for a short duration
    root.images['cvcs_logo'] = PhotoImage(file="Stuff/CVCSTitleScreenLogo.png")
    cvcs_logo_label = tk.Label(root, image=root.images['cvcs_logo'], bg='black')
    cvcs_logo_label.place(x=0, y=0, relwidth=1, relheight=1)
    root.after(3000, close_window)

root = tk.Tk()
root.title("Title Screen Logo")
root.attributes('-fullscreen', True)
root.images = {}
root.protocol("WM_DELETE_WINDOW", close_all_windows)  # Ensures clean closure

show_project_logo()
root.mainloop()