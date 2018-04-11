import math
from collections import namedtuple

import numpy as np
from PIL import Image, ImageDraw


def bounding_box(z_where, x_size):
    """This doesn't take into account interpolation, but it's close
    enough to be usable."""
    w = x_size / z_where.s
    h = x_size / z_where.s
    xtrans = -z_where.x / z_where.s * x_size / 2.
    ytrans = -z_where.y / z_where.s * x_size / 2.
    x = (x_size - w) / 2 + xtrans  # origin is top left
    y = (x_size - h) / 2 + ytrans
    return (x, y), w, h


def arr2img(arr):
    # arr is expected to be a 2d array of floats in [0,1]
    return Image.frombuffer('L', arr.shape, (arr * 255).astype(np.uint8).tostring(), 'raw', 'L', 0, 1)


def img2arr(img):
    # assumes color image
    # returns an array suitable for sending to visdom
    return np.array(img.getdata(), np.uint8).reshape(img.size + (3,)).transpose((2, 0, 1))


def colors(k):
    return [(255, 0, 0), (0, 255, 0), (0, 0, 255)][k % 3]


def draw_one(imgarr, z_arr):
    # Note that this clipping makes the visualisation somewhat
    # misleading, as it incorrectly suggests objects occlude one
    # another.
    clipped = np.clip(imgarr.detach().cpu().numpy(), 0, 1)
    img = arr2img(clipped).convert('RGB')
    draw = ImageDraw.Draw(img)
    for k, z in enumerate(z_arr):
        # It would be better to use z_pres to change the opacity of
        # the bounding boxes, but I couldn't make that work with PIL.
        # Instead this darkens the color, and skips boxes altogether
        # when z_pres==0.
        if z.pres > 0:
            (x, y), w, h = bounding_box(z, imgarr.size(0))
            color = tuple(map(lambda c: int(c * z.pres), colors(k)))
            draw.rectangle([x, y, x + w, y + h], outline=color)
    is_relaxed = any(z.pres != math.floor(z.pres) for z in z_arr)
    fmtstr = '{:.1f}' if is_relaxed else '{:.0f}'
    draw.text((0, 0), fmtstr.format(sum(z.pres for z in z_arr)), fill='white')
    return img2arr(img)


def draw_many(imgarrs, z_arr):
    # canvases is expected to be a (n,w,h) numpy array
    # z_where_arr is expected to be a list of length n
    return [draw_one(imgarr, z) for (imgarr, z) in zip(imgarrs.cpu(), z_arr)]


z_obj = namedtuple('z', 's,x,y,pres')


# Map a tensor of latents (as produced by latents_to_tensor) to a list
# of z_obj named tuples.
def tensor_to_objs(latents):
    return [[z_obj._make(step) for step in z] for z in latents]
