import os
import sys
import myutil
from PIL import Image
import numpy as np
import time
from myutil import toMatrix
from myutil import toImage
from myutil import toImage_rgb
from myutil import padding
from myutil import padding_rgb
from myutil import depadding
from myutil import view_as_window
from myutil import toGrayMatrix
from myutil import scale
from MyPIL import filter2d
from MyPIL import meanFilter

def get_min_img(mtx):

    matrix = []
    for i in range(len(mtx)):
        row = []
        for j in range(len(mtx[0])):
            row.append(min(mtx[i][j]))
        matrix.append(row)
    return matrix

def dark_channel(patches, (x, y)):

    """ get the dark channel """
    pixels = [np.min(patch) for patch in patches]
    matrix = np.array(pixels).reshape(y, x).transpose()
    return matrix


def get_A(I_dark, I):

    cnt = [[] for i in range(256)]
    for i in range(len(I_dark)):
        for j in range(len(I_dark[0])):
            cnt[I_dark[i][j]].append((i, j))
    Sum = 0
    index = 255
    Max = -1
    A = (-1, -1, -1)
    num = (len(I)) * (len(I[0])) / 1000
    while Sum < num:
        for (x, y) in cnt[index]:
            idensity = sum(I[x][y]) / 3.0
            if idensity > Max:
                Max = idensity
                A = I[x][y]
            Sum += 1
            if Sum >= num:
                break
        if Sum >= num:
            break
        index -= 1
    #A = [175 if p > 175 else p for p in A] # limit the value of A
    return A

def guided_filter(p_, I_):

    """ take guide filter for transmission tx """
    p = np.array(p_)
    I = np.array(I_)
    e = 0.005
    size = (21, 21) # size of mean filter

    mean_I = meanFilter(I, size)
    mean_P = meanFilter(p, size)
    corr_I = meanFilter(I*I, size)
    corr_IP = meanFilter(I*p, size)

    var_I = corr_I - mean_I * mean_I
    cov_IP = corr_IP - mean_I * mean_P

    a = cov_IP / (var_I + e)
    b = mean_P - a * mean_I

    mean_a = meanFilter(a, size)
    mean_b = meanFilter(b, size)

    q = mean_a * I + mean_b
    return q


def free_haze(I, size, Type, name):
    """
    I: input image
    size: size of window
    Type: 1 means using the guided filter
    name: name of the image
    """

    M = len(I)
    N = len(I[0])
    MM = M+2*(size[0]-1)
    NN = N +2*(size[1]-1)
    w = 0.95
    T = 0.1

    matrix = padding_rgb(I, (MM, NN), 255)
    min_img = get_min_img(matrix) # take min operation on rgb
    patches = myutil.view_as_window(min_img, size)
    I_dark = depadding(dark_channel(patches, (M+2*(size[0]/2), N+2*(size[1]/2))), (M, N))
    A = get_A(I_dark, I)

    for i in range(MM):
        for j in range(NN):
            min_img[i][j] = np.min(tuple(np.array(matrix[i][j])*1.0 / np.array(A)))
    patches = myutil.view_as_window(min_img, size)
    tmp = depadding(dark_channel(patches, (M+2*(size[0]/2), N+2*(size[1]/2))), (M, N))
    tx = 1 - w * np.array(tmp)     # compute the transmission
    if Type == 1:                  # guided filter
        tx = guided_filter(tx, toGrayMatrix(I))
        toImage(myutil.scale(tx)).save(name+'_tx_21.png')
    else:
        toImage(myutil.scale(tx)).save(name+'_tx.png')



    J = []
    for i in range(M):
        row = []
        for j in range(N):
            lst = []
            for k in range(3):
                lst.append(int((I[i][j][k] - A[k]) * 1.0/ (max(tx[i][j], T)) + A[k]))
            row.append(tuple(lst))
        J.append(row)
    return J


if __name__ == '__main__':


    print 'Name of Image: ',
    name = raw_input()
    print 'Type: 1 for guided filter: ',
    Type = int(raw_input())

    start = time.clock()

    input_img = Image.open(name+'.jpg')
    if input_img.mode == 'RGBA':
        input_img = myutil.toRGB(input_img)
    output_img = free_haze(toMatrix(input_img), (15, 15), Type, name)
    output_img = toImage_rgb(output_img)
    if Type == 1:
        output_img.save(name+'_21.png')
    else :
        output_img.save(name+'_output.png')

    elapsed = (time.clock() - start)
    print "time used: ", elapsed

