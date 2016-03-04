from PIL import Image
import sys
import random
import math
import numpy as np
from copy import deepcopy

def toRGB(input_img):
    """ Image --> Image """
    source = input_img.split()
    output = Image.merge('RGB', (source[0], source[1], source[2]))
    return output

def toGray(input_img):
    """ Image --> Image """
    output = Image.new('L', input_img.size)
    for h in range(output.size[1]):
        for w in range(output.size[0]):
            (r, g, b) = input_img.getpixel((w, h))
            val = 0.299 * r + 0.587 * g + 0.114 * b
            output.pixel((w, h), va)
    return output


def toMatrix(source):
    """ Image --> matrix """
    matrix = []
    print source.size
    for h in range(0, source.size[1]):
        row = []
        for w in range(0, source.size[0]):
            #print source.getpixel((w, h))
            row.append(source.getpixel((w, h)))
        matrix.append(row)
    return matrix

def toGrayMatrix(input_img):
    """ Matrix --> Matrix """
    matrix = []
    for i in range(0, len(input_img)):
        row = []
        for j in range(0, len(input_img[0])):
            (r, g, b) = input_img[i][j]
            val = 0.299 * r + 0.587 * g + 0.114 * b
            row.append(int(val))
        matrix.append(row)
    return matrix


def toImage(matrix):
    """ Matrix --> Image """
    target = Image.new("L", (len(matrix[0]), len(matrix)))
    for h in range(0, target.size[1]):
        for w in range(0, target.size[0]):
            target.putpixel((w, h), int(matrix[h][w]))
    return target

def toImage_rgb(matrix):
    """ Matrix of tuple --> Image """
    target = Image.new("RGB", (len(matrix[0]), len(matrix)))
    for h in range(target.size[1]):
        for w in range(target.size[0]):
            #print matrix[h][w]
            target.putpixel((w, h), matrix[h][w])
    return target


def scale(mtr):
    """ scale """
    matrix = deepcopy(mtr)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if math.isnan(matrix[i][j]):
                matrix[i][j] = 0
            if math.isinf(matrix[i][j]):
                matrix[i][j] = 255
    mx = np.max(matrix)
    mn = np.min(matrix)
    interval = mx - mn
    for h in range(len(matrix)):
        for w in range(len(matrix[0])):
            matrix[h][w] = int(((matrix[h][w] - mn) * 255) / interval)
    return matrix

def scale2(matr):
    """
    scale
    if pixel larger than 255, then set pixel = 255
    if pixel less than 0, then set pixel = 0
    other pixels keep unchanged
    """
    matrix = deepcopy(mtr)
    for h in range(len(matrix)):
        for w in range(len(matrix[0])):
            if (matrix[h][w] > 255):
                matrix[h][w] = 255
            if (matrix[h][w] < 0):
                matrix[h][w] = 0
            if (math.isnan(matrix[h][w])):
                matrix[h][w] = 0
    return matrix

def padding(input_img, size, p):
    """ expand the input image """
    M = len(input_img)
    N = len(input_img[0])
    stx = (size[0]-M) / 2
    sty = (size[1]-N) / 2
    matrix = []
    for x in range(size[0]):
        row = []
        for y in range(size[1]):
            if x >= stx and x < (stx+M) and y >= sty and y < (sty+N):
                row.append(input_img[x-stx][y-sty])
            else:
                row.append(p)
        matrix.append(row)
    return matrix

def padding_rgb(input_img, size, p):
    """ expand the input image """
    M = len(input_img)
    N = len(input_img[0])
    stx = (size[0]-M) / 2
    sty = (size[1]-N) / 2
    matrix = []
    for x in range(size[0]):
        row = []
        for y in range(size[1]):
            if x >= stx and x < (stx+M) and y >= sty and y < (sty+N):
                row.append(input_img[x-stx][y-sty])
            else:
                row.append((p, p, p))
        matrix.append(row)
    return matrix

def depadding(input_img, size):
    """ remove the padding """
    M = len(input_img)
    N = len(input_img[0])
    stx = (M-size[0]) / 2
    sty = (N-size[1]) / 2
    matrix = []
    for x in range(M):
        if x >= stx and x < (stx+size[0]):
            row = []
            for y in range(N):
                if y >= sty and y < (sty+size[1]):
                    row.append(input_img[x][y])
            matrix.append(row)
    return matrix

def getPatch(filename, patch):
    output_img = Image.new('L', (len(patch[0]), len(patch)))
    for h in range(len(patch)):
        for w in range(len(patch[0])):
            output_img.putpixel((w, h), patch[h][w])
    output_img.save(filename)

def view_as_window(img, patch_size):
    patches = []
    for w in range(len(img[0])-patch_size[0]+1):
        for h in range(len(img)-patch_size[1]+1):
            patch = []
            if h == 0 :
                for i in range(patch_size[1]):
                    patch.append([img[h+i][j] for j in range(w, w+patch_size[0])])
            else :
                patch += [patches[-1][i] for i in range(1, len(patches[-1]))]
                patch.append([img[h+patch_size[1]-1][w+i] for i in range(patch_size[0])])
            patches.append(patch)
    return patches
