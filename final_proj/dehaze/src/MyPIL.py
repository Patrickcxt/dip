from PIL import Image
import os
import sys
import math
import numpy as np
import myutil
from myutil import toMatrix
from myutil import toImage
from myutil import scale
import random
import matplotlib.pyplot as pil
from functools import reduce
from copy import deepcopy

def multiply(lst):
    result = 1
    for i in range(len(lst)):
        for j in range(len(lst[0])):
            result *= lst[i][j]
    return result

def getMean(mtr, pos, size):

    m = len(mtr)
    n = len(mtr[0])
    N = size[0]/2
    stx = max(0, pos[0]-N)
    sty = max(0, pos[1]-N)
    edx = min(pos[0]+N+1, m)
    edy = min(pos[1]+N+1, n)
    s = 0
    for i in range(stx, edx):
        for j in range(sty, edy):
            s += mtr[i][j]
    return s /((edx-stx)*(edy-sty))

def meanFilter(input_img, size):
    N = size[0]
    mtr = []
    for i in range(len(input_img)):
        row = []
        for j in range(len(input_img[0])):
            row.append(getMean(input_img, (i, j), size))
        mtr.append(row)
    return np.array(mtr)


def filter2d(input_img, filter_type, fsize):
    """
    filter_type == 0: arithmetic filter
    filter_type == 1:
    filter_type == 2:
    filter_type == 3:
    filter_type == 4: harmonic filter
    filter_type == 5: contraharmonic filter
    filter_type == 6: median filter
    filter_type == 7: max filter
    filter_type == 8: min filter
    filter_type == 9: geometric mean filter
    """

    M = len(input_img)
    N = len(input_img[0])
    print M, " ", N
    print (M+2*(fsize[0]-1), N+2*(fsize[1]-1))
    tmp_img = myutil.padding(input_img, (M+2*(fsize[0]-1), N+2*(fsize[1]-1)), 0)
    patches = myutil.view_as_window(tmp_img, fsize)

    tpl = []
    if filter_type == 0: # arithmetic filter
        MN = fsize[0] * fsize[1]
        for i in range(fsize[1]):
            tpl.append([1.0/MN for j in range(fsize[0])])
    elif filter_type == 1:
        tpl = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    elif filter_type == 2:
        tpl = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    elif filter_type == 3:
        tpl = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    if filter_type < 4:
        print '-----------------------------'
        pixels = [np.vdot(patch, tpl) for patch in patches]
    if filter_type == 4:
        pixels = [M*N/np.sum(1.0/np.array(patch)) for patch in patches]
    if filter_type == 5:
        pixels = [np.sum(np.array(patch)**(1.5)) / np.sum(np.array(patch)**(0.5)) for patch in patches]
    if filter_type == 6:
        pixels = [np.median(patch) for patch in patches]
    if filter_type == 7:
        pixels = [np.min(patch) for patch in patches]
    if filter_type == 8:
        pixels = [np.max(patch) for patch in patches]
    if filter_type == 9:
        pixels = [math.pow(multiply(patch), 1.0/(fsize[0]*fsize[1])) for patch in patches]


    print '-----------------------------'
    x = M + 2 * (fsize[0] / 2)
    y = N + 2 * (fsize[1] / 2)
    output = myutil.depadding(np.array(pixels).reshape(y, x).transpose(), (M, N))
    output = np.array(output)
    return output

def filter2d_rgb(input_img, filter_type, size):

    source = input_img.split()
    out = [None, None, None]
    for i in range(3):
        out[i] = filter2d(source[i], filter_type, size)

    if len(source) == 3:
        output_img = Image.merge('RGB', tuple(out))
    else:
        out.append(source[3])
        output_img = Image.merge('RGBA', tuple(out))
    return output_img



def getGauss(mean, sigma):
    x = random.random()
    y = random.random()
    theta = 2 * math.pi * x
    r = math.sqrt(-2 * math.log(y))
    z = r * math.cos(theta)
    return mean + z * sigma

def addGaussianNoise(matrix, (mean, sigma)):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = matrix[i][j] + getGauss(mean, sigma)
    matrix = myutil.scale2(matrix)
    return matrix

def addPepperSaltNoise(matrix, (p, q)):

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            rand = random.random()
            if rand < p:
                matrix[i][j] = 0
            elif rand < p+q:
                matrix[i][j] = 255
    return matrix

def addNoise(input_img, tup, flag):

    if flag == 0:
        source = input_img.split()
        out = [None, None, None]
        '''
        for i in range(3):
            out[i] = addGaussianNoise(toMatrix(source[i]), tup)
            out[i] = toImage(out[i])
        '''
        out[0] = toImage(addGaussianNoise(toMatrix(source[0]), tup))
        out[1] = out[0]
        out[2] = out[0]
        if (len(source) == 3):
            output_img = Image.merge('RGB', (r, g, b))
        if (len(source) == 4):
            out.append(source[3])
            output_img = Image.merge('RGBA', tuple(out))

    elif flag == 1:
        source = input_img.split()
        out = [None, None, None]
        '''
        for i in range(3):
            out[i] = addPepperSaltNoise(toMatrix(source[i]), tup)
            out[i] = toImage(out[i])
        '''
        out[0] = toImage(addPepperSaltNoise(toMatrix(source[0]), tup))
        out[1] = out[0]
        out[2] = out[0]
        if (len(source) == 3):
            output_img = Image.merge('RGB', tuple(out))
        if (len(source) == 4):
            out.append(source[3])
            output_img = Image.merge('RGBA', tuple(out))
    return output_img


def show_histogram(x, y, title, xl, yl):
    fig = pil.figure()
    ax = fig.add_subplot(111)
    ax.hist(x, 256, weights=y)
    pil.title(title)
    pil.xlabel(xl)
    pil.ylabel(yl)
    fig.show()

def plot_hist(input_img, title):
    x = range(256)
    cnt = [0] * 256
    for h in range(input_img.size[1]):
        for w in range(input_img.size[0]):
            pixel = input_img.getpixel((w, h))
            cnt[pixel] += 1

    p = range(256)
    MN = input_img.size[0] * input_img.size[1]
    S = 0
    for i in range(256):
        p[i] = cnt[i] * 1.0 / MN
        S += p[i]
    print 'p: ', S

    show_histogram(x, p, title, 'r', 'p(r)')
    return p



def equalize_hist(matrix):
    '''
    histogram equalization for a single
    channel or a gray-scale image
    '''
    x = range(256)
    cnt = [0] * 256
    result = cnt
    M = len(matrix)
    N = len(matrix[0])
    for h in range(M):
        for w in range(N):
            cnt[matrix[h][w]] += 1
    curSum = 0
    MN = M * N

    for i in range(256):
        curSum += cnt[i]
        result[i] = (255 * curSum) / MN

    cnt2 = range(256)
    for h in range(M):
        for w in range(N):
            matrix[h][w] = result[matrix[h][w]]
    return matrix

def equalize_hist_rgb(input_img):
    '''
    apply histogram equalization
    to process R, G, B channels separately
    for a rgb or rgba image
    '''
    source = input_img.split()
    out = [None, None, None]
    for i in range(3):
        out[i] = toImage(equalize_hist(toMatrix(source[i])))
        #plot_hist(out[i], str(i))
    if len(source) == 3:
        output_img = Image.merge('RGB', tuple(out))
    else:
        out.append(source[3])
        output_img = Image.merge('RGBA', tuple(out))
    return output_img

def average_hist(p):

    result = [0] * 256
    N = len(p)
    print N
    S = 0
    for i in range(256):
        #print p[0][i], ' ', p[1][i], ' ', p[2][i]
        result[i] = sum([p[j][i] for j in range(N)]) / N
        S += result[i]
        #print result[i]
    print S
    show_histogram(range(256), result, 'average', 'r', 'p(r)')
    return result

def equalize_hist_average(input_img):
    '''
    apply the average histogram equalization to
    process the R, G, B channels individually
    '''
    source = input_img.split()
    out = [None, None, None]
    p = []
    for i in range(3):
        p.append(plot_hist(source[i], ''))
    result = average_hist(p)
    reflect = [0] * 256
    curSum = 0
    for i in range(256):
        curSum += result[i]
        reflect[i] = 255 * curSum
        #print i, " ", reflect[i]

    for i in range(3):
        out[i] = Image.new('L', input_img.size)
        for h in range(input_img.size[1]):
            for w in range(input_img.size[0]):
                pixel = source[i].getpixel((w, h))
                out[i].putpixel((w, h), reflect[pixel])
    if len(source) == 3:
        output_img = Image.merge('RGB', tuple(out))
    else:
        out.append(source[3])
        output_img = Image.merge('RGBA', tuple(out))
    return output_img

if __name__ == '__main__':

    input_img = Image.open('photo/01.png')
    output_img = filter2d(input_img, 8, (15, 15))
    output_img.save('min.png')


