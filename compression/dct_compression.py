from PIL import Image
import numpy as np
from heapq import heappush, heappop, heapify
import scipy
from collections import defaultdict
import math
import matplotlib.pyplot as plt


def read_image(file_name):
    im = np.asarray(Image.open(file_name))
    return im[:, :, 0:3]


def rgb2ycbcr(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    return np.dstack((y, cb, cr))


def ycbcr2rgb(img):
    y = img[:, :, 0]
    cb = img[:, :, 1]
    cr = img[:, :, 2]

    r = y + 1.402 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)

    return np.dstack((r, g, b))


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def dct1(matrix): # not used because it was too slow
    m, n = matrix.shape
    dct = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ci = 1 / np.sqrt(m) if i == 0 else np.sqrt(2) / np.sqrt(m)
            cj = 1 / np.sqrt(n) if j == 0 else np.sqrt(2) / np.sqrt(n)
            sum_ = 0
            for k in range(m):
                for l in range(n):
                    sum_ += matrix[k, l] * np.cos((2 * k + 1) * i * np.pi / (2 * m)) * np.cos(
                        (2 * l + 1) * j * np.pi / (2 * n))
            dct[i][j] = ci * cj * sum_
    return dct


def idct1(matrix): # not used because it was too slow
    m, n = matrix.shape
        idct = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                sum = 0
                for l in range(n):
                    ci = 1 / np.sqrt(m) if k == 0 else np.sqrt(2) / np.sqrt(m)
                    cj = 1 / np.sqrt(n) if l == 0 else np.sqrt(2) / np.sqrt(n)
                    sum += ci * cj * matrix[k, l, chanel] * np.cos((2 * i + 1) * k * np.pi / (2 * m)) * np.cos(
                        (2 * j + 1) * l * np.pi / (2 * n))
                idct[i][j] = sum
    return idct


def dct(im):
    imsize = im.shape
    dct = np.zeros(imsize)
    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            for k in range(0, imsize[2]):
                dct[i:(i + 8), j:(j + 8), k] = dct2(im[i:(i + 8), j:(j + 8), k])
    return dct


def idct(dct):
    imsize = dct.shape
    im = np.zeros(imsize)
    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            for k in range(0, imsize[2]):
                im[i:(i + 8), j:(j + 8), k] = idct2(dct[i:(i + 8), j:(j + 8), k])
    return im.astype(np.uint8)


def symbol_probability_dict(im):
    di = {}
    m, n, t = im.shape
    for i in range(m):
        for j in range(n):
            for k in range(t):
                sign = str(im[i, j, k])
                if sign in di:
                    di[sign] += 1
                else:
                    di[sign] = 1
    return di


def huffman_tree(symbols):
    h = [[p, [s, ""]] for s, p in symbols.items()]
    heapify(h)
    while len(h) > 1:
        lo = heappop(h)
        hi = heappop(h)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(h, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heappop(h)[1:], key=lambda p: (len(p[-1]), p)))


def quantize_image(im, qmatrix):
    imsize = im.shape
    qshape = qmatrix.shape
    for i in range(0, imsize[0]):
        for j in range(0, imsize[1]):
            for k in range(0, imsize[2]):
                im[i, j, k] = round(im[i, j, k] / qmatrix[i % qshape[0], j % qshape[1]])


def dequantize_image(im, qmatrix):
    imsize = im.shape
    qshape = qmatrix.shape
    for i in range(0, imsize[0]):
        for j in range(0, imsize[1]):
            for k in range(0, imsize[2]):
                im[i, j, k] = im[i, j, k] * qmatrix[i % qshape[0], j % qshape[1]]


def encode_image(im, huffman_dict):
    encoded_image = ''
    imsize = im.shape
    for i in range(0, imsize[0]):
        for j in range(0, imsize[1]):
            for k in range(0, imsize[2]):
                encoded_image += huffman_dict[str(im[i, j, k])]
    return encoded_image


def decode_image(encoded_image, huffman_dict, imsize):
    inv_map = {v: k for k, v in huffman_dict.items()}
    decoded_image = []
    code = ''
    for bit in encoded_image:
        code += bit
        if code in inv_map:
            decoded_image.append(int(float(inv_map[code])))
            code = ''
    return np.asarray(decoded_image).reshape(imsize)



# assembling all together
quantization_matrix = np.array([[5, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])
im = read_image("image.jpg")
rgb2ycbcr(im)
qd = dct(im)
quantize_image(qd, quantization_matrix)
symbol_frequency = symbol_probability_dict(qd)
huffman_dict = huffman_tree(symbol_frequency)
encoded = encode_image(qd, huffman_dict)
decoded = decode_image(encoded, huffman_dict, im.shape)
dequantize_image(decoded, quantization_matrix)
new_im = idct(decoded)
ycbcr2rgb(new_im)
im = Image.fromarray(new_im)
im.save("dct_compressed.png")





