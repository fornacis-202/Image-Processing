from PIL import Image

im = Image.open('1665_girl_with_a_pearl_earring_sm.jpg')
pix = im.load()  # getting pixel values
width, height = im.size  # Get the width and height of the image for iterating over


def find_closest_palette_color(old_pixel):  # this implementation quantize each color chanle to two level by rounding values to 0 or 255
    r, g, b = old_pixel
    nr = 255 * round(r / 255)
    ng = 255 * round(g / 255)
    nb = 255 * round(b / 255)
    return nr, ng, nb


def pixel_multiplication_and_summation(pixel, error, constant):  # this function multiplies error by constant and add it to old pixel to generate the new pixel
    r, g, b = error
    ro, go, bo = pixel
    ro = int(r * constant + ro)
    if ro > 255:  # assigning values that are higher than 255 to 255
        ro = 255
    go = int(g * constant + go)
    if go > 255:
        go = 255
    bo = int(b * constant + bo)
    if bo > 255:
        bo = 255
    return ro, go, bo


for y in range(height):
    for x in range(width):
        old_pixel = pix[x, y]
        new_pixel = find_closest_palette_color(old_pixel)
        pix[x,y] = new_pixel
        error = (old_pixel[0] - new_pixel[0], old_pixel[1] - new_pixel[1], old_pixel[2] - new_pixel[2])
        if x + 1 < width:  # checking if x and y got out of the image width and height
            pix[x + 1, y] = pixel_multiplication_and_summation(pix[x + 1, y], error, 7 / 16)  # adding the specific porportion of the error to the specific pixel
        if x - 1 >= 0 and y + 1 < height:
            pix[x - 1, y + 1] = pixel_multiplication_and_summation(pix[x - 1, y + 1], error, 3 / 16)
        if y + 1 < height:
            pix[x, y + 1] = pixel_multiplication_and_summation(pix[x, y + 1], error, 5 / 16)
        if y + 1 < height and x + 1 < width:
            pix[x + 1, y + 1] = pixel_multiplication_and_summation(pix[x + 1, y + 1], error, 1 / 16)

im.save('FS-dithered.png')
