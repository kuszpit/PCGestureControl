from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController
from src.computer_controller.log import get_logger
from typing import Tuple
import time



class InputController:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InputController, cls).__new__(cls)
            cls._instance._initialized = False
            
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.logger = get_logger(__name__)
        self._initialized = True

    def move_cursor_to(self, x: int, y: int):
        self.mouse.position = (x, y)
        self.logger.info(f"Moved cursor to ({x}, {y})")


    def get_cursor_position(self) -> Tuple[int, int]:
        x, y = self.mouse.position
        self.logger.info(f"Current cursor position: ({x}, {y})")
        
        return (x, y)


    def click(self, left=True):
        if left:
            self.mouse.click(Button.left)
            self.logger.info("Left mouse button clicked")
            
            return
        
        self.mouse.click(Button.right)
        self.logger.info("Right mouse button clicked")
	
	
    def scroll_up(self, steps: int = 1):
        self.mouse.scroll(0, steps)
        self.logger.info(f"Scrolled up {steps} steps")


    def scroll_down(self, steps: int = 1):
        self.mouse.scroll(0, -steps)
        self.logger.info(f"Scrolled down {steps} steps")

