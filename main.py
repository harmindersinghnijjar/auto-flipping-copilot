import random
import time
import logging
import os
from collections import defaultdict
import numpy as np
import pyautogui
from PIL import ImageGrab
import keyboard
import wind_mouse_movement
import pygetwindow as gw
import cv2

# ==============================
# Configuration Parameters
# ==============================

CONFIG = {
    'COLOR_TOLERANCE': 20,
    'MATCHING_THRESHOLD': {
        'RED': 0.05,  # 5% of pixels need to match
        'BLUE': 0.05,
        'COLLECT': 0.05,
        'INVENTORY': 0.05,
    },
    'CLICK_RANDOMNESS': 10,  # Pixels around center to click
    'SLEEP_DURATION': {
        'MIN': 0.9,  # Increased for debugging
        'MAX': 1.3,  # Increased for debugging
    },
    'EXIT_KEY': 'q',
}

# ==============================
# Setup Logging
# ==============================

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format=('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - '
            '%(funcName)s - %(message)s'),
    handlers=[
        logging.FileHandler('slot_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a directory for screenshots if it doesn't exist
os.makedirs('screenshots', exist_ok=True)

# ==============================
# Define Bounding Boxes
# ==============================

slots = {"slot1": (46, 286 - 175, 149, 369 - 175), "slot2": (163, 286 - 175, 266, 369 - 175), "slot3": (280, 286 - 175, 383, 369 - 175), "slot4": (397, 286 - 175, 500, 369 - 175), "slot5": (46, 406 - 175, 149, 489 - 175), "slot6": (163, 406 - 175, 266, 489 - 175), "slot7": (280, 406 - 175, 383, 489 - 175), "slot8": (397, 406 - 175, 500, 489 - 175)}
buy_slots = {
    "slot1": (51, 328 - 175, 94, 369 - 175),
    "slot2": (168, 328 - 175, 210, 369 - 175),
    "slot3": (285, 328 - 175, 327, 369 - 175),
    "slot4": (402, 328 - 175, 444, 369 - 175),
    "slot5": (51, 448 - 175, 93, 489 - 175),
    "slot6": (168, 448 - 175, 210, 489 - 175),
    "slot7": (285, 448 - 175, 327, 489 - 175),
    "slot8": (402, 448 - 175, 444, 489 - 175),
}
inv_slot_dict = {'slot1': (591, 415 - 175, 625, 446 - 175), 'slot2': (633, 415 - 175, 667, 446 - 175), 'slot3': (675, 415 - 175, 709, 446 - 175), 'slot4': (717, 415 - 175, 751, 446 - 175), 'slot5': (590, 451 - 175, 625, 482 - 175), 'slot6': (632, 451 - 175, 667, 482 - 175), 'slot7': (674, 451 - 175, 709, 482 - 175), 'slot8': (716, 451 - 175, 751, 482 - 175)}
collect_finished_items_dict = {"slot1": (429, 260 - 175, 509, 277 - 175)}
buttons = {
    "collect_finished_items": (429, 260 - 175, 509, 277 - 175),
    "confirm_offer": (201, 470 - 175, 350, 505 - 175),
    "abort_offer": (368, 469 - 175, 382, 480 - 175),
    # "collect_coins": (612, 447, 662, 489),
    "back_arrow": (59, 484 - 175, 78, 492 - 175),
    # 'suggested_item_to_buy': (329, 594, 581, 633),  # No longer used
    # "suggested_quantity": (43, 628, 304, 634),
    # "suggested_price": (38, 624, 331, 634),
    "custom_quantity": (216, 402 - 175, 247, 423 - 175),
    "custom_price": (387, 402 - 175, 416, 421 - 175),
    "all_quantity": (175, 402 - 175, 205, 421 - 175),
    # "collect_items_after_abort": (
    #     677,
    #     446,
    #     732,
    #     493,
    # ),  # Coordinates for collecting items after abort
}

COLORS = {
    "RED_HIGHLIGHT": (167, 53, 21),
    "BLUE_HIGHLIGHT": (45, 98, 122),
    "COLLECT_HIGHLIGHT": (45, 98, 122),
    "INVENTORY_HIGHLIGHT": [(45, 98, 122)],
}


def move_runelite_upperleft():
    """
    Activates and moves the RuneLite window to the upper-left corner (0, 0).

    Ensures that the RuneLite window is the active window, setting it to the top-left corner.

    Raises:
        Exception: If the RuneLite window is not found.
    """
    try:
        win = gw.getWindowsWithTitle("RuneLite")[0]
        win.activate()
        logger.debug("Activated RuneLite window.")
        time.sleep(5)  # Allow time for the window to activate

        # Re-fetch the window after activation to ensure it's the right one
        win = gw.getWindowsWithTitle("RuneLite")[0]
        win.topleft = (0, 0)
        logger.info("Moved RuneLite window to the upper-left corner.")
    except IndexError:
        logger.error("RuneLite window not found.")
        raise Exception("RuneLite window not found.")


# ==============================
# Slot Manager Class
# ==============================


class SlotManager:
    def __init__(
        self,
        config,
        slots,
        buy_slots,
        inv_slot_dict,
        collect_finished_items_dict,
        buttons,
        colors,
    ):
        self.config = config
        self.slots = slots
        self.buy_slots = buy_slots
        self.inv_slot_dict = inv_slot_dict
        self.collect_finished_items_dict = collect_finished_items_dict
        self.buttons = buttons
        self.colors = colors
        self.state = defaultdict(bool)

    def select_custom_quantity(self):
        """
        Select custom quantity by typing 'E' and pressing 'Enter'.
        """
        logger.debug("Entered select_custom_quantity")
        logger.info("Selecting custom quantity...")
        # Click on the custom quantity field
        self.click_with_randomness(self.buttons["custom_quantity"])
        self.random_sleep()
        # Type 'E' and press enter
        pyautogui.typewrite("E")
        self.random_sleep()
        pyautogui.press("enter")
        self.random_sleep()

    def select_custom_price(self):
        """
        Select custom price by typing 'E' and pressing 'Enter'.
        """
        logger.debug("Entered select_custom_price")
        logger.info("Selecting custom price...")
        # Click on the custom price field
        self.click_with_randomness(self.buttons["custom_price"])
        self.random_sleep()
        # Type 'E' and press enter
        pyautogui.typewrite("E")
        self.random_sleep()
        pyautogui.press("enter")
        self.random_sleep()

    def select_suggested_item_to_buy(self):
        """
        Select the suggested item to buy.

        Instead of clicking, wait and press 'Enter' to auto-populate the item.
        """
        logger.debug("Entered select_suggested_item_to_buy")
        logger.info("Selecting suggested item to buy...")
        # Instead of clicking, wait and press 'Enter'
        wait_time = random.uniform(1.0, 2.0)
        logger.debug(f"Waiting for {wait_time:.2f} seconds before pressing 'Enter'")
        time.sleep(wait_time)
        pyautogui.press("enter")
        logger.info("Pressed 'Enter' to auto-populate the item.")
        self.random_sleep()

    def is_color_close(self, color, target_color, tolerance):
        """
        Check if a color is within tolerance of a target color.

        Args:
            color (tuple): The color to check.
            target_color (tuple): The target color to compare against.
            tolerance (int): The tolerance value.

        Returns:
            bool: True if the color is within tolerance, False otherwise.
        """
        logger.info(f"Color: {color}, Target Color: {target_color}")
        return all(
            abs(int(c) - int(t)) <= tolerance for c, t in zip(color, target_color)
        )

    def capture_screenshot(self, bbox, filename=None):
        """
        Capture a screenshot of the specified bounding box.

        Args:
            bbox (tuple): The bounding box coordinates.
            filename (str, optional): The filename to save the screenshot.

        Returns:
            numpy.ndarray: The screenshot as a NumPy array.
        """
        try:
            screenshot = ImageGrab.grab(bbox=bbox)
            screenshot_np = np.array(screenshot)
            if filename:
                screenshot.save(os.path.join("screenshots", filename))
            return screenshot_np
        except Exception as e:
            logger.error(f"Failed to capture screenshot for bbox {bbox}: {e}")
            return None

    def random_sleep(self, min_time=None, max_time=None):
        """
        Sleep for a random duration within configured limits.

        Args:
            min_time (float, optional): Minimum sleep time.
            max_time (float, optional): Maximum sleep time.
        """
        if min_time is None:
            min_time = self.config["SLEEP_DURATION"]["MIN"]
        if max_time is None:
            max_time = self.config["SLEEP_DURATION"]["MAX"]
        duration = random.uniform(min_time, max_time)
        logger.debug(f"Sleeping for {duration:.2f} seconds")
        time.sleep(duration)

    def click_with_randomness(self, bbox):
        """
        Click within the bounding box with random offset for human-like behavior.

        Args:
            bbox (tuple): The bounding box coordinates.
        """
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            quadrant_x = (x2 - x1) // 4
            quadrant_y = (y2 - y1) // 4
            offset_x = random.randint(-quadrant_x, quadrant_x)
            offset_y = random.randint(-quadrant_y, quadrant_y)
            click_x = center_x + offset_x
            click_y = center_y + offset_y
            wind_mouse_movement.wind_mouse(
                pyautogui.position().x,
                pyautogui.position().y,
                click_x,
                click_y,
                move_mouse=pyautogui.moveTo,
            )
            pyautogui.click(click_x, click_y)
            logger.info(f"Clicked at ({click_x}, {click_y})")
        except Exception as e:
            logger.error(f"Failed to click within bbox {bbox}: {e}")

    def is_highlighted(self, bbox, target_color, tolerance, threshold, identifier=""):
        """
        Determine if the area defined by bbox is highlighted with the target color.

        Args:
            bbox (tuple): The bounding box coordinates.
            target_color (tuple): The target RGB color.
            tolerance (int): The color tolerance.
            threshold (float): The matching threshold percentage.
            identifier (str, optional): Identifier for logging and screenshot naming.

        Returns:
            bool: True if the area is highlighted, False otherwise.
        """
        screenshot_np = self.capture_screenshot(bbox, filename=f"{identifier}.png")
        if screenshot_np is None:
            return False
        diff = np.abs(screenshot_np[:, :, :3] - target_color)
        matches = np.all(diff <= tolerance, axis=2)
        percentage = np.sum(matches) / matches.size
        return percentage >= threshold

    def perform_action(self, action_func, *args, **kwargs):
        """
        Execute a function with error handling.

        Args:
            action_func (callable): The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        """
        try:
            action_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error performing action {action_func.__name__}: {e}")

    def get_average_color(self, image_path, corner="upper_right", area_size=(10, 10)):
        """
        Get the average color from a specific corner of the image.

        Args:
            image_path (str): Path to the image file.
            corner (str): Corner to compute the average color ('upper_right', 'upper_left', etc.).
            area_size (tuple): Size of the area to calculate the average color (height, width).

        Returns:
            tuple: Average color as (R, G, B).
        """
        logger.debug(f"Getting average color for {image_path}, corner: {corner}")
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Image not found at path: {image_path}")
                return None

            h, w = area_size
            if corner == "upper_right":
                region = image[:h, -w:]
            elif corner == "upper_left":
                region = image[:h, :w]
            elif corner == "lower_right":
                region = image[-h:, -w:]
            elif corner == "lower_left":
                region = image[-h:, :w]
            else:
                logger.error(f"Invalid corner specified: {corner}")
                return None

            average_color_bgr = np.mean(region, axis=(0, 1))
            average_color_rgb = average_color_bgr[::-1]  # Convert BGR to RGB
            return tuple(map(int, average_color_rgb))
        except Exception as e:
            logger.error(f"Error getting average color for {image_path}: {e}")
            return None

    def test_match_with_average_color(self, image_path, target_color, tolerance):
        """
        Test if the average color from an image matches the target color within tolerance.

        Args:
            image_path (str): Path to the image file.
            target_color (tuple): Target color as (R, G, B).
            tolerance (int): Tolerance for color matching.

        Returns:
            bool: True if the color matches, False otherwise.
        """
        logger.debug(f"Testing match for {image_path} against color {target_color}")
        average_color = self.get_average_color(
            image_path, corner="upper_right", area_size=(10, 10)
        )
        if average_color is None:
            return False

        logger.info(f"Average color: {average_color}, Target color: {target_color}")
        return self.is_color_close(average_color, target_color, tolerance)

    def handle_collect_finished_items(self):
        logger.info("Handling collect finished items...")
        for slot_num, bbox in self.collect_finished_items_dict.items():
            highlighted = self.is_highlighted(
                bbox,
                self.colors["COLLECT_HIGHLIGHT"],
                self.config["COLOR_TOLERANCE"],
                self.config["MATCHING_THRESHOLD"]["COLLECT"],
                identifier=f"collect_finished_items_{slot_num}",
            )
            if highlighted:
                self.click_with_randomness(self.buttons["collect_finished_items"])
                self.random_sleep()

    def handle_abort_slots(self):
        """
        Handle actions when an abort offer is highlighted.

        Steps:
            - Check if an abort offer is highlighted in the slot.
            - Click the slot if highlighted.
            - Check if the abort offer button is highlighted.
            - Click the abort offer button if highlighted.
            - Go back to the main screen.
        """
        logger.info("Handling abort slots...")

        for slot_num, bbox in self.slots.items():
            logger.debug(f"Checking slot {slot_num} for abort highlight...")

            # Check if the slot is highlighted for abort
            highlighted = self.is_highlighted(
                bbox,
                self.colors["RED_HIGHLIGHT"],
                self.config["COLOR_TOLERANCE"],
                self.config["MATCHING_THRESHOLD"]["RED"],
                identifier=f"slot_{slot_num}",
            )
            if highlighted:
                logger.info(f"Slot {slot_num} is highlighted for abort.")
                self.click_with_randomness(bbox)
                self.random_sleep()
                
                # Click on the abort offer button
                self.perform_action(self.click_with_randomness, self.buttons["abort_offer"])
                logger.info("Returning to the main screen.")
                self.click_with_randomness(self.buttons["back_arrow"])
                self.random_sleep()
                
            else:
                logger.debug(f"Slot {slot_num} is not highlighted for abort.")


    def handle_buy_slots(self):
        """
        Handle actions when a buy slot is highlighted.

        Steps:
            - Check if the buy slot is highlighted.
            - Select the suggested item to buy.
            - Check for custom quantity and set it if highlighted.
            - Check for custom price and set it if highlighted.
            - Confirm the offer if highlighted.
            - If no confirmation is detected, return to the main screen.
        """
        logger.info("Handling buy slots...")

        for slot_num, bbox in self.buy_slots.items():
            # Check if the buy slot is highlighted
            highlighted = self.is_highlighted(
                bbox,
                self.colors["BLUE_HIGHLIGHT"],
                self.config["COLOR_TOLERANCE"],
                self.config["MATCHING_THRESHOLD"]["BLUE"],
                identifier=f"buy_slot_{slot_num}",
            )
            if highlighted:
                logger.info(f"Buy slot {slot_num} is highlighted.")
                self.click_with_randomness(bbox)
                self.random_sleep()

                # Select the suggested item to buy
                self.select_suggested_item_to_buy()
                self.random_sleep()

                # Check for custom quantity
                highlighted_quantity = self.is_highlighted(
                    self.buttons["custom_quantity"],
                    self.colors["BLUE_HIGHLIGHT"],
                    self.config["COLOR_TOLERANCE"],
                    self.config["MATCHING_THRESHOLD"]["BLUE"],
                    identifier=f"buy_custom_quantity",
                )
                if highlighted_quantity:
                    logger.info("Custom quantity button is highlighted.")
                    self.select_custom_quantity()
                    self.random_sleep()

                # Check for custom price
                highlighted_price = self.is_highlighted(
                    self.buttons["custom_price"],
                    self.colors["BLUE_HIGHLIGHT"],
                    self.config["COLOR_TOLERANCE"],
                    self.config["MATCHING_THRESHOLD"]["BLUE"],
                    identifier=f"buy_custom_price",
                )
                if highlighted_price:
                    logger.info("Custom price button is highlighted.")
                    self.select_custom_price()
                    self.random_sleep()

                # Confirm the offer if highlighted
                highlighted_confirm = self.is_highlighted(
                    self.buttons["confirm_offer"],
                    self.colors["BLUE_HIGHLIGHT"],
                    self.config["COLOR_TOLERANCE"],
                    self.config["MATCHING_THRESHOLD"]["BLUE"],
                    identifier=f"buy_confirm_offer",
                )
                if highlighted_confirm:
                    logger.info("Confirm offer button is highlighted.")
                    self.click_with_randomness(self.buttons["confirm_offer"])
                    self.random_sleep()
                else:
                    logger.info("No confirm offer detected. Returning to the main screen.")
                    self.click_with_randomness(self.buttons["back_arrow"])
                    self.random_sleep()
            else:
                logger.debug(f"Buy slot {slot_num} is not highlighted.")

    def handle_inventory_slots(self):
        """
        Handle actions when an inventory slot is highlighted.

        Steps:
            - Check if an inventory slot is highlighted.
            - Click on the highlighted slot.
            - Check if all_quantity is highlighted.
            - Click on all_quantity if highlighted.
            - Check if custom_price is highlighted.
            - Click on custom_price if highlighted.
            - Select custom price by typing 'E' and pressing 'Enter'.
            - Check if confirm_offer is highlighted.
            - Click on confirm_offer if highlighted.
            - If no confirmation is detected, return to the main screen.
        """
        logger.info("Handling inventory slots...")
        for slot_num, bbox in self.inv_slot_dict.items():
            highlighted = self.is_highlighted(
                bbox,
                self.colors["INVENTORY_HIGHLIGHT"][0],
                self.config["COLOR_TOLERANCE"],
                self.config["MATCHING_THRESHOLD"]["INVENTORY"],
                identifier=f"inv_slot_{slot_num}",
            )
            if highlighted:
                logger.info(f"Inventory slot {slot_num} is highlighted.")
                self.click_with_randomness(bbox)
                self.random_sleep()

                # Check if all_quantity is highlighted
                highlighted_all_quantity = self.is_highlighted(
                    self.buttons["all_quantity"],
                    self.colors["BLUE_HIGHLIGHT"],  # Assuming all_quantity uses BLUE_HIGHLIGHT
                    self.config["COLOR_TOLERANCE"],
                    self.config["MATCHING_THRESHOLD"]["BLUE"],
                    identifier="inv_all_quantity",
                )
                if highlighted_all_quantity:
                    logger.info("All quantity button is highlighted.")
                    self.click_with_randomness(self.buttons["all_quantity"])
                    self.random_sleep()
                else:
                    logger.info("All quantity button is not highlighted. Skipping selection.")

                # Check if custom_price is highlighted
                highlighted_custom_price = self.is_highlighted(
                    self.buttons["custom_price"],
                    self.colors["BLUE_HIGHLIGHT"],
                    self.config["COLOR_TOLERANCE"],
                    self.config["MATCHING_THRESHOLD"]["BLUE"],
                    identifier="inv_custom_price",
                )
                if highlighted_custom_price:
                    logger.info("Custom price button is highlighted.")
                    self.select_custom_price()
                    self.random_sleep()
                else:
                    logger.info("Custom price button is not highlighted. Skipping selection.")

                # Check if confirm_offer is highlighted
                highlighted_confirm = self.is_highlighted(
                    self.buttons["confirm_offer"],
                    self.colors["BLUE_HIGHLIGHT"],
                    self.config["COLOR_TOLERANCE"],
                    self.config["MATCHING_THRESHOLD"]["BLUE"],
                    identifier="inv_confirm_offer",
                )
                if highlighted_confirm:
                    logger.info("Confirm offer button is highlighted.")
                    self.click_with_randomness(self.buttons["confirm_offer"])
                    self.random_sleep()
                else:
                    logger.info("No confirm offer detected. Returning to the main screen.")
                    self.click_with_randomness(self.buttons["back_arrow"])
                    self.random_sleep()
            else:
                logger.debug(f"Inventory slot {slot_num} is not highlighted.")

    def run(self):
        logger.info("Starting Slot Manager...")
        try:
            while True:
                if random.random() < 0.05:  # Simulate human-like idle time
                    idle_time = random.uniform(5, 15)
                    logger.info(f"Simulating idle period for {idle_time:.2f} seconds.")
                    time.sleep(idle_time)

                actions = [
                    self.handle_collect_finished_items,
                    self.handle_abort_slots,
                    self.handle_buy_slots,
                    self.handle_inventory_slots,
                ]
                random.shuffle(actions)
                for action in actions:
                    action()

                if keyboard.is_pressed(self.config["EXIT_KEY"]):
                    logger.info("Exit key pressed. Exiting Slot Manager.")
                    break

                self.random_sleep()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Exiting Slot Manager.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            logger.info("Slot Manager has stopped.")


# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    move_runelite_upperleft()
    slot_manager = SlotManager(
        CONFIG, slots, buy_slots, inv_slot_dict, collect_finished_items_dict, buttons, COLORS
    )

    slot_manager.run()
