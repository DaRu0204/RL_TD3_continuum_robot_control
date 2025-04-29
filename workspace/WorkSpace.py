import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
dataset_path = os.path.join(base_dir, "dataset", "Dataset-Actions-Positions.txt")
data = pd.read_csv(dataset_path, header=None)

# Select rows containing x, y, z
x = data[3]
y = data[4]
z = data[5]

# Compute central axis with constant x and z, varying y
center_x = np.mean(x)
center_z = np.mean(z)
y_min, y_max = np.min(y), np.max(y)

# Define line points along the central axis
line_y = np.linspace(y_min, y_max, 100)
line_x = np.full_like(line_y, center_x)
line_z = np.full_like(line_y, center_z)

# Function to generate a circle in the X-Z plane
def generate_circle(radius, center_x, center_z, center_y, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = center_x + radius * np.cos(theta)
    circle_z = center_z + radius * np.sin(theta)
    circle_y = np.full_like(circle_x, center_y)
    return circle_x, circle_y, circle_z

# Function to generate a rectangle in the X-Z plane
def generate_rectangle(width, height, center_x, center_z, center_y):
    half_w, half_h = width / 2, height / 2
    rect_x = [center_x - half_w, center_x + half_w, center_x + half_w, center_x - half_w, center_x - half_w]
    rect_z = [center_z - half_h, center_z - half_h, center_z + half_h, center_z + half_h, center_z - half_h]
    rect_y = np.full_like(rect_x, center_y)
    return rect_x, rect_y, rect_z


# Function to generate an 8/3 star in the X-Z plane
def generate_star(radius, center_x, center_z, center_y, tilt_angle=0):
    angles = np.linspace(0, 2 * np.pi, 17)[:-1]  # 16 points
    inner_radius = radius / 2.618  # Golden ratio approximation
    star_x = []
    star_z = []
    for i in range(16):
        r = radius if i % 2 == 0 else inner_radius
        star_x.append(r * np.cos(angles[i]))
        star_z.append(r * np.sin(angles[i]))
    
    star_x.append(star_x[0])  # Closing the star
    star_z.append(star_z[0])
    
    # Apply tilt rotation around the x-axis
    tilt_rad = np.radians(tilt_angle)
    star_y = np.zeros_like(star_x)  # Center star in the X-Z plane
    rotated_y = star_y * np.cos(tilt_rad) - np.array(star_z) * np.sin(tilt_rad)
    rotated_z = star_y * np.sin(tilt_rad) + np.array(star_z) * np.cos(tilt_rad)
    
    # Translate to correct position
    star_x = np.array(star_x) + center_x
    rotated_y = np.array(rotated_y) + center_y
    rotated_z = np.array(rotated_z) + center_z
    
    return star_x, rotated_y, rotated_z

# Function to plot everything
def plot_scene(show_points=True, show_axis=True, show_circle=True, show_rectangle=True, show_star=True, 
               circle_radius=10, rect_width=20, rect_height=10, star_radius=15, star_tilt=0,
               circle_y=None, rect_y=None, star_y=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    if show_points:
        ax.scatter(x, y, z, c='y', marker='o', alpha=0.03, label='Dataset Points')
    
    if show_axis:
        ax.plot(line_x, line_y, line_z, c='r', linewidth=2, label='Continuum Robot')
    
    if show_circle:
        if circle_y is None:
            circle_y = (y_min + y_max) / 2
        circle_x, circle_y, circle_z = generate_circle(circle_radius, center_x, center_z, circle_y)
        ax.plot(circle_x, circle_y, circle_z, c='g', linewidth=2, label='Circle')
    
    if show_rectangle:
        if rect_y is None:
            rect_y = (y_min + y_max) / 2
        rect_x, rect_y, rect_z = generate_rectangle(rect_width, rect_height, center_x, center_z, rect_y)
        ax.plot(rect_x, rect_y, rect_z, c='b', linewidth=2, label='Rectangle')
    
    if show_star:
        if star_y is None:
            star_y = (y_min + y_max) / 2
        star_x, star_y, star_z = generate_star(star_radius, center_x, center_z, star_y, star_tilt)
        ax.plot(star_x, star_y, star_z, c='m', linewidth=2, label='8/3 Star')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_aspect("equal")
    ax.set_title('Dataset Visualization')
    ax.legend()
    
    plt.show()
    
# Main function
def main():
    plot_scene(show_points=True, show_axis=True, show_circle=True, show_rectangle=True, show_star=True,
               circle_radius=35, circle_y=-135, rect_width=60, rect_height=35, rect_y=-135,
               star_radius=25, star_tilt=-10, star_y=-135)
    print(center_x)
    print(center_z)
    print((y_min + y_max) / 2)

if __name__ == "__main__":
    main()