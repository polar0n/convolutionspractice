from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


img = np.array(Image.open('images/frame_gray.png'))

H = 1/52
X = -1/37
line_kernel = np.array([
    [0, 0, 0, 0, H, H, H, 0, 0, 0, 0],
    [0, 0, H, H, H, H, H, H, H, 0, 0],
    [0, H, H, H, X, X, X, H, H, H, 0],
    [0, H, H, X, X, X, X, X, H, H, 0],
    [H, H, X, X, X, X, X, X, X, H, H],
    [H, H, X, X, X, X, X, X, X, H, H],
    [H, H, X, X, X, X, X, X, X, H, H],
    [0, H, H, X, X, X, X, X, H, H, 0],
    [0, H, H, H, X, X, X, H, H, H, 0],
    [0, 0, H, H, H, H, H, H, H, 0, 0],
    [0, 0, 0, 0, H, H, H, 0, 0, 0, 0]
])

line_conv = fftconvolve(img, line_kernel)

lines_img = np.zeros(
    tuple(line_conv.shape) + (3,),
    dtype=np.uint8
)

pixels = line_conv.astype(np.int16).copy()
np.place(pixels, pixels < 0, 0)
np.place(pixels, pixels > np.ceil(pixels.flatten().mean()), 255)
np.place(pixels, pixels <= np.ceil(pixels.flatten().mean()), 0)
pixels = pixels.astype(np.uint8)

lines_img[:,:,0] = pixels
lines_img[:,:,1] = pixels
lines_img[:,:,2] = pixels


def unit_circle_vectorized(r, threshold=1):
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    return (np.abs(dists-r)<threshold).astype(int)


def gen_circles(r):
    circle_kernel = unit_circle_vectorized(r).astype(np.float32)
    counts = np.unique(circle_kernel, return_counts=True)
    np.place(circle_kernel, circle_kernel == counts[0][0], 0)
    np.place(circle_kernel, circle_kernel == counts[0][1], 1/counts[1][1])

    circle_convolution = fftconvolve(lines_img[:,:,0], circle_kernel)

    circles_img = np.zeros(
        tuple(circle_convolution.shape) + (3,),
        dtype=np.uint8
    )

    pixels = circle_convolution.astype(np.int16).copy()
    np.place(pixels, pixels < 0, 0)
    np.place(pixels, pixels > np.ceil(pixels.flatten().mean())+200, 255)
    np.place(pixels, pixels <= np.ceil(pixels.flatten().mean())+200, 0)
    pixels = pixels.astype(np.uint8)

    circles_img[:,:,0] = pixels
    circles_img[:,:,1] = pixels
    circles_img[:,:,2] = pixels
    
    Image.fromarray(circles_img).show()


for i in range(4, 20+4):
    gen_circles(i)
