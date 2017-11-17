from pdb import set_trace as bb
import cairocffi
from numpy import genfromtxt
from os.path import splitext
import torch
from torch import Tensor as T
from torch import zeros as ZT
from torch import ones as OT
from torch.autograd import Variable as V
import torch.nn.functional as F
import numpy as np
from visdom import Visdom
from utils import Vector2  # ,BBOX2, Viewport
from functools import partial

getbuffer = np.getbuffer if hasattr(np, "getbuffer") else memoryview
two_pi = 2*np.pi


# pytorch object holding image data
# that can be rendered to screen if required
# or comparison can be made to target/other
class PtImg2D():

    def __init__(self, pt_img=None, img_path=None, width=None, height=None, empty_white=True):
        # paint the image regardless of making empty
        assert (pt_img is not None) or (img_path is not None) or (width is not None and height is not None),\
            "must provide either and image or a width/height"

        # store pt_img directly
        if pt_img is not None:
            self.pt_img = pt_img
        else:
            # are we loading from an image?
            # nope
            if img_path is None:
                # create all ones (white)
                self.pt_img = V(OT([4, height, width])) \
                    if empty_white \
                    else V(ZT([4, height, width]))
            else:
                # load into a ImageSurface, then read into numpy -> pytorch var
                self.pt_img = surface_to_pt(img_to_surface(img_path))

        # height and width pulled from shape
        self.height, self.width = self.pt_img.data.shape[1:]

    # compare img versus sobel tgt
    # https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/utils.js#L75
    def weighted_percent_same_bin(self, target_img, sobel_pt_img, weight):
        assert self.width == target_img.width and self.height == target_img.height and \
            self.width == sobel_pt_img.data.shape[2] and self.height == sobel_pt_img.data.shape[1], \
            'weighted_percent_same_bin: image dimensions do not match!'

        # conver RGBA to grayscale
        this_gs, other_gs = rgba_to_grayscale(
            self.pt_img).data, rgba_to_grayscale(target_img.pt_img).data

        # TODO: Do we want to support this only across a single channel?
        # this is currently across all channels
        # original code uses only the first channel of info
        this_empty, other_empty = this_gs == 1.0, other_gs == 1.0

        # when both are empty or not empty
        where_equal = this_empty == other_empty

        # create an empty set with ones (see line w=... below for explanation)
        sim_tensor = OT([1, self.width, self.height])

        # index into non empties and set equal to sobel tgt using weight
        # var w = otherEmpty ? 1 : flatWeight +
        # (1-flatWeight)*sobelImg.data[i/4];
        sim_tensor[~other_empty] = weight + \
            (1 - weight) * sobel_pt_img.data[~other_empty]

        # sum up all of the values where they're equal
        sim_sum = sim_tensor[where_equal].sum()

        # sum up all weighted combos
        weight_sum = sim_tensor.sum()

        # send back sim/weight
        return sim_sum / weight_sum


def rgba_to_grayscale(pt_img):
    # https://stackoverflow.com/questions/42516203/converting-rgba-image-to-grayscale-golang
    # 0.299 * R +  0.587 * G + 0.114 * B
    return .299 * pt_img[0:1] + .587 * pt_img[1:2] + .114 * pt_img[2:3]


def img_to_surface(file_name):
    return cairocffi.ImageSurface.create_from_png(file_name)


def all_white(width, height, channels=4):
    return np.ones([channels, width, height], dtype="uint8")*255


def empty_surface(width, height, is_white=False):
    data = None if not is_white else getbuffer(all_white(width, height).reshape(-1))
    return cairocffi.ImageSurface(cairocffi.FORMAT_ARGB32,
                                  width,
                                  height,
                                  data=data)


def np_to_surface(np_img):
    np_uint_data = (255*np_img).astype("uint8")
    return cairocffi.ImageSurface(cairocffi.FORMAT_ARGB32,
                                  np_img.shape[2],
                                  np_img.shape[1],
                                  data=getbuffer(np_uint_data.reshape(-1)))


# convert surface into [0,1] floats
def surface_to_np(img_surface):
    np_surface = np.frombuffer(img_surface.get_data(), np.uint8)
    # reshape according to the surface size
    np_surface = np_surface.reshape([img_surface.get_height(),
                                     img_surface.get_width(),
                                     4])
    # finally, transpose to be C, H, W standard pytorch convention
    return np.transpose(np_surface, [2, 0, 1])


# convert from pytorch variable to numpy, then to a surface object
def pt_to_surface(pt_img):
    return np_to_surface(pt_img.data.numpy())


# return an image between 0,1 as float variable
def surface_to_pt(img_surface):
    np_surface = surface_to_np(img_surface).astype("float32") / 255.
    # make sure it's in
    return V(torch.from_numpy(np_surface))


# pull from:
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/index.js
def load_target(target_file):

    base_tgt_file = splitext(target_file)[0]

    # convert our intial coords to numpy array
    # contains both the start position and directional information for an img
    coord_info = genfromtxt(base_tgt_file + '.txt',
                            delimiter=' ').astype(float)

    # load and paint onto a cairo surface
    target = PtImg2D(img_path=base_tgt_file + '.png')

    # add some properties
    # like starting position and direction
    target.start_pos = Vector2(*coord_info[0])
    target.start_dir = Vector2(*coord_info[1])

    # send back our image object
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
SOBEL_XY = torch.cat([SOBEL_X_FILTER, SOBEL_Y_FILTER],
                     dim=0).expand(2, 4, 3, 3)
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
    # resulting in: [C, H, W]
    return torch.sqrt(img_xy[0, :1]**2 + img_xy[0, 1:]**2)


def make_gradient_weight_similarity_fct(edge_weight):

    # wrap var in closure
    flat_weight = 1. / edge_weight

    def weighted_percent(img, tgt_img):
        # take our image objects and then create sobel
        sobel_pt_img = get_sobel(tgt_img.pt_img)

        # return the weighted percent by similarity
        return img.weighted_percent_same_bin(tgt_img, sobel_pt_img, flat_weight)

    # return a function that will use sobel filter to
    # compare img and target
    return weighted_percent


# apply similarity metric to completely white image
def baseline_similarity(target_img, similarity_fct):
    assert isinstance(target_img, PtImg2D), \
        "Assuming baseline image is PtImg2D"

    # create our empty image filled with white (all 1s)
    white_img = PtImg2D(None, None, target_img.width,
                        target_img.height, empty_white=True)

    # then measure similarity with our target
    return similarity_fct(white_img, target_img)


# normalize similarity against the baseline (similarity compared to white
# background)
def normalized_similarity(img, target_img, similarity_fct):
    assert isinstance(img, PtImg2D) and isinstance(target_img, PtImg2D), \
        "Assuming image and target are PtImg2D"

    sim_comparison = similarity_fct(img, target_img)

    # haven't calculated baseline? do it here.
    if not hasattr(target_img, "baseline"):
        target_img.baseline = baseline_similarity(target_img, similarity_fct)

    # remove baseline from comparison, and baseline from perfect sim score (1)
    # then divide to get normalized score
    return (sim_comparison - target_img.baseline) / (1. - target_img.baseline)


# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/render.js#L296
def render_branch(context, branch):
    # brown branch
    context.set_source_rgb(89/255., 58/255., 26/255.)
    context.new_path()
    context.set_line_width(branch["width"].data[0])
    context.set_line_cap(cairocffi.LINE_CAP_ROUND)
    context.move_to(branch["start"].x, branch["start"].y)
    context.line_to(branch["end"].x, branch["end"].y)
    context.stroke()
    context.close_path()


# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/render.js#L304
def render_leaf(context, leaf):
    context.save()
    context.set_source_rgb(68/255., 106/255., 57/255.)
    context.translate(leaf["center"].x, leaf["center"].y)
    context.rotate(leaf["angle"].data[0])
    context.scale(leaf["length"].data[0], leaf["width"].data[0])
    context.new_path()
    context.arc(0, 0, 0.5, 0, two_pi)
    context.fill()
    context.close_path()
    context.restore()


# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/render.js#L315
def render_flower(context, flower):
    context.set_source_rgb(194/255., 75/255., 119/255.)
    context.new_path()
    context.arc(flower["center"].x, flower["center"].y, flower["radius"].data[0], 0, two_pi)
    context.fill()
    context.close_path()


def render_type(context, render_obj):
    print("render: {}".format(render_obj))
    geo_type = render_obj["type"].lower()
    rfct = globals()['render_{}'.format(geo_type)]
    rfct(context, render_obj[geo_type])


def subset_type(geo_states, geo_type):
    return [geo for geo in geo_states if geo["type"] == geo_type]


# // Render update
# most logic matches this function:
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/render.js#L334
def render_update(full_geo, viewport, print_render=False):

    # get our init state with some info
    init_state, geo_states = full_geo[0], full_geo[1:]
    img_size = init_state["img_size"]
    start_viewport = init_state["viewport"]

    # create our empty canvas
    img_surface = empty_surface(*img_size, is_white=True)

    # draw context
    context = cairocffi.Context(img_surface)

    # set context with initial viewport and scaling
    vwidth, vheight = start_viewport.xmax - start_viewport.xmin, start_viewport.ymax - start_viewport.ymin
    context.scale(img_size[0]/vwidth, img_size[1]/vheight)
    context.translate(-start_viewport.xmin, -start_viewport.ymin)

    # all states in ascending order
    # we'll subset out specific types and render accordingly
    all_states = sorted(geo_states, key=lambda x: x["n"])

    # context render fct
    render_type_fct = partial(render_type, context)
    subset_fct = partial(subset_type, all_states)

    # render branches first
    list(map(render_type_fct, subset_fct('branch')))
    list(map(render_type_fct, subset_fct('leaf')))
    list(map(render_type_fct, subset_fct('flower')))

    # return to our tensor
    pt_img = surface_to_pt(img_surface)

    if print_render:
        vis = Visdom(env='paint')
        vis.image(pt_img.data.numpy())

    # adjust viewport? not currently
    return PtImg2D(pt_img=pt_img), viewport


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-tgt', "--target_file", type=str, required=True,
                        help='target to match')

    args = parser.parse_args()

    # load our target object
    tgt = load_target(args.target_file)

    # apply sobel to the R of the rgba
    sobel_img = get_sobel(tgt.pt_img)

    # basic sim function, send in edge weight for closure
    sim_fct = make_gradient_weight_similarity_fct(1.5)

    # get baseline similarity
    baseline = baseline_similarity(tgt, sim_fct)

    # should be perfect similarity
    perfect_sim = normalized_similarity(tgt, tgt, sim_fct)

    # invert the image
    inv_img = PtImg2D(pt_img=(1.0 - tgt.pt_img))

    # then just get number for inverted sim
    fake_sim = normalized_similarity(inv_img, tgt, sim_fct)

    # print out test render
    print("Test render finish: \n\
           base: {}, \n\
           perfect sim (should be 1): {}, \n\
           inverted sim: {}".format(baseline, perfect_sim, fake_sim))
