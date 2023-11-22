from PIL import Image

im = Image.open('s.jpg')
pix = im.load()  # getting pixel values
width, height = im.size  # Get the width and height of the image for iterating over

# making the image grayscale
for x in range(width):
    for y in range(height):
        r, g, b = pix[x, y]
        gray = (r + g + b) // 3 # averaging over the R,G,B chanel
        pix[x, y] = (gray, gray, gray) # setting the average value for each chanel to obtain a gray color

im.save('grayscale.png')  # saving the grayscale image

dithering_matrix = [[0, 8, 2, 10],
                    [12, 4, 14, 5],
                    [3, 11, 1, 9],
                    [15, 7, 13, 5]]
l = 256 / 17.0 # computing the constant by which each grayscale value between 0-255 must be divided to obtain a value between 0-15

# similar to the pseudocode mentioned in the slides
for x in range(width):
    for y in range(height):
        i = x % 4
        j = y % 4
        value = pix[x, y][0] / l
        if value > dithering_matrix[j][i]:
            pix[x, y] = (255, 255, 255)  # setting to black
        else:
            pix[x, y] = (0, 0, 0)  # setting to white
im.save('o_dithered.png')
