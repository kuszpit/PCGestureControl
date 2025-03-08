from enum import Enum
import tkinter as tk

from src.computer_controller.input_controller import InputController

class States(Enum):
    cursor = 1
    left_click = 2
    right_click = 3
    scroll = 4


class GestureProcessor:
    state = 0

    def __init__(self):
        self.prev_position = None
        self.input_controller = InputController()

    def process(self, gesture, positions):
        gesture = gesture.strip()
        
        print(gesture)

        if gesture == "palm_up":
            self.move_cursor(positions)
        elif gesture == "fist":
            self.left_click(positions)
        elif gesture == "span":
            self.right_click(positions)
        elif gesture == "palm_sideways":
            self.scroll(positions)


    def move_cursor(self, positions):
        landmarks = positions.multi_hand_landmarks[0].landmark
        wrist_landmark = landmarks[0]
        
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        
        screen_x = wrist_landmark.x * width
        screen_y = wrist_landmark.y * height

        self.input_controller.move_cursor_to(screen_x, screen_y)
     

    def left_click(self, positions):
        self.input_controller.click(True)
        
        
    def right_click(self, positions):
        self.input_controller.click(False)


    def scroll(self, positions):
        landmarks = positions.multi_hand_landmarks[0].landmark
        wrist_landmark = landmarks[0]

        y_change = self.prev_position.y - wrist_landmark.y

        self.input_controller.scroll_up(y_change)

        


