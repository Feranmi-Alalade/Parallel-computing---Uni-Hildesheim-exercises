import sys

row = 0 # to keep tracks of the rows

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    # split into pixels
    pixels = line.split()

    pixel_list = []
    for pixel in pixels:
        # Split into channels and append to pixel list
        channels = pixel.split(",")
        pixel_list.append(channels)

    for col, pixel in enumerate(pixel_list):
        # to divide the image into 10x10 patches
        patch_row = row//10
        patch_col = col//10

        # The red, green and blue channels in the pixel
        red = pixel[0]
        green = pixel[1]
        blue = pixel[2]

        print(f"{patch_row},{patch_col}\t{row},{col},{red},{green},{blue}")

    row += 1


