import war
import tkinter as tk
from tkinter import PhotoImage

war_label = None
second_window = None

def on_war_click():
    war_label.config(text="Playing War")
    war_label.destroy() # war_label not being destoryed here?
    war.open_camera()

def close_window():
    root.destroy()
    open_second_window()

def open_second_window():
    # Create a new top-level window
    global second_window
    second_window = tk.Toplevel()
    second_window.title("Second Window")

    # Configure the background color of the second window
    second_window.configure(bg='black')

    # Set the geometry of the second window
    window_width = 400
    window_height = 300
    screen_width = second_window.winfo_screenwidth()
    screen_height = second_window.winfo_screenheight()
    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = (screen_height - window_height) // 2
    second_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    # Create buttons in the new window
    button1 = tk.Button(second_window, text="War", command=option1_click)
    button1.pack(pady=10)

    button2 = tk.Button(second_window, text="Go Fish", command=option2_click)
    button2.pack(pady=10)

    button3 = tk.Button(second_window, text="Crazy 4's", command=option3_click)
    button3.pack(pady=10)

    button4 = tk.Button(second_window, text="Black Jack", command=option4_click)
    button4.pack(pady=10)

    button5 = tk.Button(second_window, text="Solitare", command=option5_click)
    button5.pack(pady=10)

def option1_click():
    second_window.destroy()
    global war_label
    war_option = tk.Toplevel()
    war_option.title("Rules of War")

    window_width = 500
    window_height = 300
    screen_width = war_option.winfo_screenwidth()
    screen_height = war_option.winfo_screenheight()
    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = (screen_height - window_height) // 2

    war_option.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    text = "The Game of War \n Take the deck and split it evenly between the players \n Make sure to do this without looking at the face of the cards \n After dealing the cards, both players play the card on top of their card pile \n"
    text = text + "After both players play their top card, compare the two cards \n Which ever card's rank is higher wins \n If the ranks are the same, then you go to war \n"
    war_label = tk.Label(war_option, text=text)

    # Create a button
    war_button = tk.Button(war_option, text="Play Game", command=on_war_click)

    war_label.pack(padx=20, pady=20)
    war_button.pack(side=tk.BOTTOM, padx=20, pady=20)

def option2_click():
    print("Go Fish selected")

def option3_click():
    print("Crazy 4's selected")

def option4_click():
    print("Black Jack selected")

def option5_click():
    print("Solitare selected")

# Create the main window
root = tk.Tk()
root.title("Image Background GUI")

# Load the image
image_path = "Logo.png"  # Renamed the image to Logo for convenience
background_image = PhotoImage(file=image_path)

# Get the image dimensions
image_width = background_image.width()
image_height = background_image.height()

# Set the window size to match the image size
root.geometry("500x400")

# Create a Canvas widget to display the image
canvas = tk.Canvas(root, width=500, height=400)
canvas.pack()

# Place the image on the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

root.after(2000, close_window)
# Start the Tkinter event loop
root.mainloop()
