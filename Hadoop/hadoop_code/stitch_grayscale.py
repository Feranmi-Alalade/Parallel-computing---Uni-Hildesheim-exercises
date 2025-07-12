import ast
import numpy as np
import matplotlib.pyplot as plt

pixels = []

with open('output_grayscale.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        # split into patch id and values containing the rows, columns and grayscale
        patch_id, values = line.split("\t")

        values = ast.literal_eval(values)

        pixels.extend(values)

rows = [pixel[0] for pixel in pixels]
cols = [pixel[1] for pixel in pixels]
grayscales = [pixel[2] for pixel in pixels]

# get the maximum number of rows and columns to know the image size
n_image_rows = max(rows) + 1
n_image_cols = max(cols) + 1

# define buffer with the size
image = np.zeros((n_image_rows, n_image_cols), dtype=np.int32)

for row, col, grayscale in pixels:
    image[row,col] = grayscale

plt.imshow(image, cmap='gray')

plt.show()


