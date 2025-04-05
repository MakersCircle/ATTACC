import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

video_name = '001484'
video_folder = Path(__file__).parent / video_name

# Define paths
original_path = video_folder / f'{video_name}.mp4'
overlay_path = video_folder / f'{video_name}_detection.mp4'
depth_path = video_folder / f'{video_name}_depth.mp4'
prob_toa_path = video_folder / f'{video_name}_prob_toa.npz'
output_path = video_folder / f'{video_name}_demo.gif'

# Load probabilities
data = np.load(prob_toa_path, allow_pickle=True)
probabilities = data['probabilities']
toa = data['toa']
tta = data['tta']

num_frames = len(probabilities)

# Open video captures
cap_orig = cv2.VideoCapture(original_path)
cap_overlay = cv2.VideoCapture(overlay_path)
cap_depth = cv2.VideoCapture(depth_path)

# Get frame dimensions
width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the layout (2 rows, 3 columns)
fig = plt.figure(figsize=(12, 6))

# First row: 3 images (videos)
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)

# Second row: 1 graph spanning all 3 columns
ax4 = plt.subplot(2, 1, 2)  # This spans the whole second row

# Titles for the first row
ax1.set_title('Original Video')
ax2.set_title('Object Detection Video')
ax3.set_title('Depth Video')

# Hide axes for a clean look (for video axes)
for ax in [ax1, ax2, ax3]:
    ax.axis('off')

# Create image objects for videos
img1 = ax1.imshow(np.zeros((height, width, 3), dtype=np.uint8))
img2 = ax2.imshow(np.zeros((height, width, 3), dtype=np.uint8))
img3 = ax3.imshow(np.zeros((height, width, 3), dtype=np.uint8))

# Set up the probability graph
line, = ax4.plot([], [], 'b-', label='Probabilities')
ax4.set_xlim(0, num_frames)
ax4.set_ylim(0, 1)
ax4.set_xlabel("Frame")
ax4.set_ylabel("Probability")
ax4.set_title('Accident Probability')

# Additional plot elements
ax4.axhline(y=0.5, color='g', linestyle=':', label='Threshold = 0.5')  # Horizontal threshold line
if toa is not None:
    ax4.axvline(x=toa, color='r', linestyle='--', label=f'TOA = {toa}')  # Vertical TOA line
if tta is not None:
    ax4.axvline(x=toa - tta, color='y', linestyle='--', label=f'Anticipated at {toa - tta}')

ax4.text(1, 0, "github.com/MakersCircle/ATTACC", transform=ax4.transAxes, ha='right', va='bottom', fontsize=10, color='grey', fontweight='light')

ax4.legend()
ax4.grid(True)

# Update function for animation
def update(frame_idx):
    ret1, frame1 = cap_orig.read()
    ret2, frame2 = cap_overlay.read()
    ret3, frame3 = cap_depth.read()
    if not (ret1 and ret2 and ret3): return

    # Convert BGR to RGB for display
    img1.set_data(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    img2.set_data(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    img3.set_data(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB))

    # Update the probability graph
    line.set_data(np.arange(frame_idx + 1), probabilities[:frame_idx + 1])
    return img1, img2, img3, line

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=False)


ani.save(filename=output_path, writer='pillow')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
