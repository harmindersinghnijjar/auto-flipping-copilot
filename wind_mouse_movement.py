# Constants
from ctypes import windll
import random
import time
import pyautogui as gui
import numpy as np

# Define constants for square roots and golden ratio
sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)
golden_ratio = (1 + sqrt3) / 2


def wind_mouse(
    start_x,
    start_y,
    dest_x,
    dest_y,
    G_0=12,  # Increased gravitational pull for faster movement
    W_0=2,  # Reduced wind influence for a more direct path
    M_0=20,  # Larger max step size
    D_0=10,  # Shorter distance before damping begins
    move_mouse=lambda x, y: None,
):
    """
    WindMouse algorithm optimized to reach the destination faster and more directly.
    """
    current_x, current_y = start_x, start_y
    v_x = v_y = W_x = W_y = 0

    while (dist := np.hypot(dest_x - start_x, dest_y - start_y)) >= 1:
        W_mag = min(W_0, dist / D_0)

        if dist >= D_0:
            # More direct movement with less random influence
            W_x = W_x / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
            W_y = W_y / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
        else:
            # Reduced randomness when close to target
            W_x /= sqrt3
            W_y /= sqrt3
            M_0 = max(3, M_0 / sqrt5)  # Decrease step size near target

        # Stronger gravitational pull towards target
        v_x += W_x + G_0 * (dest_x - start_x) / dist
        v_y += W_y + G_0 * (dest_y - start_y) / dist
        v_mag = np.hypot(v_x, v_y)

        # Clip the velocity to ensure smoother movement
        if v_mag > M_0:
            v_clip = M_0 / 1.5 + np.random.random() * M_0 / 3
            v_x = (v_x / v_mag) * v_clip
            v_y = (v_y / v_mag) * v_clip

        start_x += v_x
        start_y += v_y
        move_x = int(np.round(start_x))
        move_y = int(np.round(start_y))

        # Move only if there is a position change
        if current_x != move_x or current_y != move_y:
            move_mouse(current_x := move_x, current_y := move_y)
            time.sleep(random.uniform(0.004, 0.008))  # Reduced delay

    return current_x, current_y


def mouse_movement(x, y, duration=random.uniform(0.05, 0.08)):
    """
    Moves the mouse to (x, y) with reduced overshoot and increased speed.
    """
    points = []
    cur_x, cur_y = gui.position()
    g = random.randrange(10, 15)  # Increased gravitational pull
    w = random.randrange(1, 3)  # Reduced wind influence for more direct path
    m = random.randrange(15, 25)  # Larger max step size
    d = random.randrange(5, 10)  # Distance before damping

    # Generate points using the optimized WindMouse function
    wind_mouse(
        cur_x, cur_y, x, y, g, w, m, d, move_mouse=lambda x, y: points.append([x, y])
    )

    # Apply faster playback
    for i in points:
        x, y = i
        gui.moveTo(x, y)
        time.sleep((duration / len(points)) * 0.5 * random.uniform(0.9, 1.1))
