import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define bounderies of the workspace
x_min, x_max = -70.63, -18.33
y_min, y_max = -135, -100
z_min, z_max = 247.45, 299.76

# Load dataset
dataset_path = "workspace_point_dataset.txt"
data = pd.read_csv(dataset_path, header=None)

# Select rows containing x, y, z
x = data[3]
y = data[4]
z = data[5]

# Filter points that are outside defined workspace
filtered_data = data[(x >= x_min) & (x <= x_max) & 
                     (y >= y_min) & (y <= y_max) & 
                     (z >= z_min) & (z <= z_max)]
filtered_x = filtered_data[3]
filtered_y = filtered_data[4]
filtered_z = filtered_data[5]

# Vizualsation 1: All points and workspace
fig = plt.figure(figsize=(12, 6))

# Subplot 1: All points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z, c='b', marker='o', label='All Dataset Points')

# Vizualisation of workspace (workspace cube)
corners_x = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
corners_y = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
corners_z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

for i in range(4):
    # Lower base
    ax1.plot([corners_x[i], corners_x[(i+1)%4]], 
             [corners_y[i], corners_y[(i+1)%4]], 
             [corners_z[i], corners_z[(i+1)%4]], color='r')
    # Upper base
    ax1.plot([corners_x[i+4], corners_x[(i+1)%4+4]], 
             [corners_y[i+4], corners_y[(i+1)%4+4]], 
             [corners_z[i+4], corners_z[(i+1)%4+4]], color='r')
    # vertical edges
    ax1.plot([corners_x[i], corners_x[i+4]], 
             [corners_y[i], corners_y[i+4]], 
             [corners_z[i], corners_z[i+4]], color='r')

ax1.set_xlabel('X Coordinate')
ax1.set_ylabel('Y Coordinate')
ax1.set_zlabel('Z Coordinate')
ax1.set_title('All Points with Workspace Cube')
ax1.legend()

# Subplot 2: Points inside the workspace
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(filtered_x, filtered_y, filtered_z, c='g', marker='o', label='Points in Workspace')

# Vizualisation of workspace (workspace cube)
for i in range(4):
    # Lower base
    ax2.plot([corners_x[i], corners_x[(i+1)%4]], 
             [corners_y[i], corners_y[(i+1)%4]], 
             [corners_z[i], corners_z[(i+1)%4]], color='r')
    # Upper base
    ax2.plot([corners_x[i+4], corners_x[(i+1)%4+4]], 
             [corners_y[i+4], corners_y[(i+1)%4+4]], 
             [corners_z[i+4], corners_z[(i+1)%4+4]], color='r')
    # Vertical edges
    ax2.plot([corners_x[i], corners_x[i+4]], 
             [corners_y[i], corners_y[i+4]], 
             [corners_z[i], corners_z[i+4]], color='r')

ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.set_zlabel('Z Coordinate')
ax2.set_title('Points Inside Workspace Cube')
ax2.legend()

plt.tight_layout()
plt.show()

# Write workspace bounderies
print(f"Workspace boundaries:")
print(f"X-axis: {x_min:.2f} to {x_max:.2f}")
print(f"Y-axis: {y_min:.2f} to {y_max:.2f}")
print(f"Z-axis: {z_min:.2f} to {z_max:.2f}")

# Write number of points in the workspace
print(f"Number of points in workspace: {len(filtered_data)}")
