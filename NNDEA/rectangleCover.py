import matplotlib.pyplot as plt
import random

# Define the dimensions of the large rectangle
large_rect_width = 487
large_rect_height = 210

# Define the minimum and maximum length for the smaller rectangles
min_length = 35
max_length = 45

# Create a function to generate a random size for the smaller rectangles
def generate_rectangles(large_width, large_height, min_len, max_len):
    rectangles = []
    x = 0
    y = 0
    while y < large_height:
        while x < large_width:
            width = random.randint(min_len, max_len)
            height = random.randint(min_len, max_len)
            if x + width <= large_width and y + height <= large_height:
                rectangles.append((x, y, width, height))
                x += width
            else:
                break
        x = 0
        y += height
    return rectangles

# Generate the rectangles
rectangles = generate_rectangles(large_rect_width, large_rect_height, min_length, max_length)

# Plot the rectangles
fig, ax = plt.subplots()
for rect in rectangles:
    x, y, width, height = rect
    ax.add_patch(plt.Rectangle((x, y), width, height, fill=None, edgecolor='black'))

ax.set_xlim(0, large_rect_width)
ax.set_ylim(0, large_rect_height)
ax.set_aspect('equal')
plt.gca().invert_yaxis()
plt.show()

