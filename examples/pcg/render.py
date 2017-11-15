from pdb import set_trace as bb
import cairocffi
from numpy import genfromtxt
from os.path import splitext
import torch
from torch import Tensor as T
from torch.autograd import Variable as V
import torch.nn.functional as F
import numpy as np


# basic vector additions and lerping
class Vector2():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "{:0.5},{:0.5}".format(self.x, self.y)

    def add(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def clone(self):
        return Vector2(self.x, self.y)

    # linear interpolate between two points us and them
    def lerp(self, other, zero_one):
        self.x = zero_one * self.x + (1 - zero_one) * other.x
        self.y = zero_one * self.y + (1 - zero_one) * other.y
        return self

    # obs multiply x,y by scalar
    def scalar_mult(self, mult):
        self.x = self.x * mult
        self.y = self.y * mult
        return self


class BBOX2():
    def __init__(self, v2_min, v2_max):
        self.min = v2_min
        self.max = v2_max

    def clone(self):
        return BBOX2(self.min.clone(), self.max.clone())

    # overwrites bbox with combined union -- clone to avoid issues
    def union(self, other):
        print("warning, bbox union not verified. \
                This print statement should annoy you.")
        self.min = Vector2(min(self.min.x, other.min.x),
                           min(self.min.y, other.min.y))
        self.max = Vector2(max(self.max.x, other.max.x),
                           max(self.max.y, other.max.y))
        return self


# pull from:
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/index.js
def load_target(target_file):

    base_tgt_file = splitext(target_file)[0]

    # convert our intial coords to numpy array
    # contains both the start position and directional information for an img
    coord_info = genfromtxt(base_tgt_file + '.txt', delimiter=' ').astype(float)

    # load and paint onto a cairo surface
    tgt_surface = cairocffi.ImageSurface.create_from_png(base_tgt_file + '.png')

    # create our basic target
    target = {
        "image": tgt_surface,
        "baseline": None,
        "tensor": None,
        "startPos": Vector2(*coord_info[0]),
        "startDir": Vector2(*coord_info[1])
    }

    return target


# function getSobel(img) {
# 	var sobelImg = img.__sobel;
# 	if (img.__sobel === undefined) {
# 		img.__sobel = Sobel.sobel(img.toTensor(0, 1));
# 		sobelImg = img.__sobel;
# 	}
# 	return sobelImg;
# }

# Horizontal and vertical filters
SOBEL_X_FILTER = V(T([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]]),
                   requires_grad=False).view(1, 1, 3, 3)

SOBEL_Y_FILTER = V(T([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]]),
                   requires_grad=False).view(1, 1, 3, 3)


# concat on out channes
# format for convolution weights is [Cout, Cin, K, K]
# http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
SOBEL_XY = torch.cat([SOBEL_X_FILTER, SOBEL_Y_FILTER], dim=0)
BIAS = V(torch.zeros(SOBEL_XY.data.shape[0]), requires_grad=False)


# apply sobel to target
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/sobel.js#L23
def get_sobel(pt_img):
    # if H,W,4 convert to 1,H,W,4
    if pt_img.dim() == 3:
        pt_img = pt_img.unsqueeze(0)

    # convolve the image with the sobel filters
    img_xy = F.conv2d(pt_img, SOBEL_XY, BIAS, stride=1, padding=1)

    # take the distance between the two
    # keep the channel dimension == 1, remove the batch dim
    # resulting in: [1, H, W]
    return torch.sqrt(img_xy[0, 0:1]**2 + img_xy[0, 1:2]**2)


# convert surface into [0,1] floats
def surface_to_np(img_surface):
    np_surface = np.frombuffer(img_surface.get_data(), np.uint8)
    # reshape according to the surface size
    np_surface = np_surface.reshape([img_surface.get_height(),
                                     img_surface.get_width(),
                                     4])
    # finally, transpose to be C, H, W standard pytorch convention
    return np.transpose(np_surface, [2, 0, 1])


# return an image between 0,1 as float variable
def surface_to_pt(img_surface):
    np_surface = surface_to_np(img_surface).astype("float32")/255.
    # make sure it's in
    return V(torch.from_numpy(np_surface))


def make_gradient_weight_similarity(ed):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-tgt', "--target_file", type=str, required=True,
                        help='target to match')

    args = parser.parse_args()

    tgt = load_target(args.target_file)

    # convert to variable -- no grads
    pt_img = surface_to_pt(tgt['image'])

    # apply sobel to the R of the rgba
    sobel_img = get_sobel(pt_img[0:1, :, :])

    bb()

