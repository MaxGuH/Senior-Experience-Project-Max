import tkinter as tk
import crazy4 # Not made yet
import war

war_label = None

def on_war_click():
    war_label.config(text="Playing War")
    root.destroy()
    war.open_camera() #

def on_button_click(title):
    print(f"Selected title: {title}")

    if title == "Crazy 4's":
        print("Ligma")
        #crazy4.open_camera() # If the method doesn't work, make sure both files are saved first
    elif title == "War":
        global war_label
        top_level = tk.Toplevel(root)  # Use Toplevel instead of Tk
        top_level.title("Rules of War")

        window_width = 500
        window_height = 300
        screen_width = top_level.winfo_screenwidth()
        screen_height = top_level.winfo_screenheight()
        x_coordinate = (screen_width - window_width) // 2
        y_coordinate = (screen_height - window_height) // 2

        top_level.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")


        text = "The Game of War \n Take the deck and split it evenly between the players \n Make sure to do this without looking at the face of the cards \n After dealing the cards, both players play the card on top of their card pile \n"
        text = text + "After both players play their top card, compare the two cards \n Which ever card's rank is higher wins \n If the ranks are the same, then you go to war \n"
        war_label = tk.Label(top_level, text= text)

        # Create a button
        war_button = tk.Button(top_level, text="Play Game", command=on_war_click)
        

        war_label.pack(padx=20, pady=20)
        war_button.pack(side=tk.BOTTOM, padx=20, pady=20)

# List of titles
titles = ["Crazy 4's", "Go Fish", "Black Jack", "Solitare", "War"]

# Create the main window
root = tk.Tk()
root.title("Title Select Screen")

root.configure(bg='black')

window_width = 400
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Create and place buttons for each title
for title in titles:
    button = tk.Button(root, text=title, command=lambda t=title: on_button_click(t), bg="lightgray")
    button.pack(pady=10)

# Run the main loop
root.mainloop()
