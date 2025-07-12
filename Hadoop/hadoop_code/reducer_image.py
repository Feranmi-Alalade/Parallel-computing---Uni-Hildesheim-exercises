import sys

current_key = None
pixels = []

def convert_to_grayscale(red, green, blue):
    """
    This function takes the three color channels
    and converts to gray scale
    """
    return float((0.2989*red) + (0.587*green) + (0.114*blue))

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    key, value = line.split("\t")

    row, col, red, green, blue = value.split(",")
    row = int(row)
    col = int(col)

    # convert to integer
    red = int(red)
    green = int(green)
    blue = int(blue)

    grayscale = convert_to_grayscale(red,green,blue)

    if key == current_key:
        
        pixels.append((row,col,grayscale))

    else:
        if current_key is not None:
            print(f"{current_key}\t{pixels}")
        current_key = key
        pixels = [(row,col,grayscale)]

if current_key is not None:
    print(f"{current_key}\t{pixels}")