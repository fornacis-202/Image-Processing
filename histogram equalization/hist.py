from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def gray_scale(pix, width, height):  # step 1
    # making the image grayscale
    for x in range(width):
        for y in range(height):
            r, g, b, ma = pix[x, y]
            gray = round(0.299 * r + 0.587 * g + 0.114 * b)  # or we could simply average over r,g and b
            pix[x, y] = (gray, gray, gray)  # setting the average value for each chanel to obtain a gray color


def histogram(pix, width, height):  # step 2
    s = np.zeros(256)
    for x in range(width):
        for y in range(height):
            g = pix[x, y][0]
            s[g] = s[g] + 1
    return s


def cumulative(s):  # step 3
    cs = np.cumsum(s)
    return cs


def create_map(cumulative, width, height):  # step 4
    mmap = ((256 - 1) / (width * height)) * cumulative
    mmap = np.round(mmap)
    return mmap


def apply_map(mmap, pix, width, height):  # step 5
    for x in range(width):
        for y in range(height):
            new_value = int(mmap[pix[x, y][0]])
            pix[x, y] = (new_value, new_value, new_value)


im = Image.open('image.png')
pix = im.load()  # getting pixel values
width, height = im.size
gray_scale(pix, width, height)
h = histogram(pix, width, height)
cum = cumulative(h)
my_map = create_map(cum, width, height)
apply_map(my_map, pix, width, height)
im.save("new.png")

# h_a = histogram(pix, width, height)
# cum_a = cumulative(h_a)
# plt.plot(np.arange(0,256),cum_a)
# plt.show()
