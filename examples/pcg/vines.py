# Author: Paul Szerlip Converted from Original JS code:
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/wppl/vines.wppl
from pdb import set_trace as bb
import numpy as np
import torch
import pyro.distributions as dist
import pyro
from collections import defaultdict
from utils import VT, VTA, BBOX2, Vector2, Viewport
from render import normalized_similarity, load_target, render_update
from render import make_gradient_weight_similarity_fct
from visdom import Visdom
# ----------------------------------------------------------------------------
# Globals / constants


def norm2world(p, viewport):
    return Vector2(	viewport.xmin + p.x * (viewport.xmax - viewport.xmin),
                    viewport.ymin + p.y * (viewport.ymax - viewport.ymin))


def polar2rect(r, theta):
    return Vector2(r * torch.cos(theta), r * torch.sin(theta))

# ----------------------------------------------------------------------------
# Factor encouraging similarity to target image


# pretty easy, extend yourself to the start and end of branch obj
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/utils.js#L337
def bbox_branch(new_branch):
    bbox = BBOX2()
    bbox.expand_by_point(new_branch["start"])
    bbox.expand_by_point(new_branch["end"])
    return bbox


def pivot(p, sin_v, cos_v, c):
    # rotate point around a center
    return Vector2(cos_v*p.x + sin_v*p.y + c.x, sin_v*p.x - cos_v*p.y + c.y)


# replacing this fct
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/utils.js#L343
def bbox_leaf(new_leaf):
    w2 = new_leaf["width"]/2
    l2 = new_leaf["length"]/2
    p0 = Vector2(-w2, -l2)
    p1 = Vector2(w2, -l2)
    p2 = Vector2(-w2, l2)
    p3 = Vector2(w2, l2)
    # just grab float value not tensor val
    f_sin = torch.sin(new_leaf["angle"]).data[0]
    f_cos = torch.cos(new_leaf["angle"]).data[0]
    center = new_leaf["center"]
    # is this pivot stuff a bug, it says p0, p0, p0, p0?
    # https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/utils.js#L364
    p0 = pivot(p0, f_sin, f_cos, center)
    p1 = pivot(p1, f_sin, f_cos, center)
    p2 = pivot(p2, f_sin, f_cos, center)
    p3 = pivot(p3, f_sin, f_cos, center)
    # expand our bounding box world
    box = BBOX2()
    box.expand_by_point(p0)
    box.expand_by_point(p1)
    box.expand_by_point(p2)
    box.expand_by_point(p3)
    return box


# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/js/utils.js#L376
def bbox_flower(new_flower):
    f_center = new_flower["center"]
    radius = new_flower["radius"].data[0]
    # get flower min/max, and use those as the bounding box
    flower_min = Vector2(f_center.x - radius, f_center.y - radius)
    flower_max = Vector2(f_center.x + radius, f_center.y + radius)

    # send back bounding box using the min/max
    return BBOX2(flower_min, flower_max)


def bbox_for_type(type_name, type_obj):
    if type_name == "flower":
        return bbox_flower(type_obj)
    elif type_name == "branch":
        return bbox_branch(type_obj)
    elif type_name == "leaf":
        return bbox_leaf(type_obj)
    else:
        raise NotImplementedError(
            "Unknown bbox for type: {}".format(type_name))


# Basically Gaussian log-likelihood, without the constant factor
def makescore(val, target, tightness):
    diff = val - target
    return - (diff**2) / (tightness**2)


def norm_prob(probs):
    return probs / probs.sum()

# holds our procedural object


class ProceduralVine():

    def __init__(self, initial_state, bbox, viewport, target, target_factor):
        # TODO, real type here
        self.target = target
        self.viewport = viewport
        self.bbox = bbox
        self.target_factor = target_factor
        self.terminated = False

        # start angle is mu,sigma for sample -- replace is with a pyro sample
        initial_state['angle'] = self._gaussian(
            'start_angle', initial_state['angle'][0],
            initial_state['angle'][1])

        # set the state as an array
        self.geo_state = [initial_state]
        self.last_geo = None
        self.last_f = None

    def _flip(self, name, p):
        return pyro.sample(name, dist.bernoulli, p).data[0]

    def _discrete3(self, name, ps, vs=None):
        return pyro.sample(name, dist.categorical, ps, vs=vs, one_hot=False)

    def _gaussian(self, name, mu, sigma):
        return pyro.sample(name, dist.normal, mu, sigma)

    def add_pyro_factor(self, site_name, val):
        # TODO: Check validity -- this is approximating the factor statement in
        # webppl with a bernoulli observe
        logit_input = val

        # factor hack!
        if self.last_f is not None:
            logit_input = val - self.last_f

        self.last_f = val

        # factor definitely happened this will add an obs to the graph and
        # bernoulli likelihood == f^whatever*(1-f)^0 == f*log_p
        pyro.observe(site_name, dist.bernoulli,
                     logits=logit_input, obs=VT([1]))

    # generically add the type to our state that will in turn be rendered and
    # then measured against the target image we'll add that factor to any
    # relevant sample sites in the model
    def add_type(self, site_name, type_name, cur_state, new_type_obj):
        geo = {
            "type": type_name,
            "next": self.last_geo,
            "parent": cur_state["prev_branch"],
            "n": self.last_geo["n"] + 1 if self.last_geo else 1
        }
        # add type object to the geo for later rendering info
        geo[type_name] = new_type_obj

        # create union of our current bounding box + new object added
        self.bbox = self.bbox.clone().union(bbox_for_type(type_name,
                                                          new_type_obj))
        self.geo_state.append(geo)

        # get target factor from choices
        f, self.viewport = self.target_factor(
            self.geo_state, self.bbox, self.viewport, self.target)

        # need to add factor to previous sample sites
        self.add_pyro_factor(site_name, f)

        # mark the last geo object
        self.last_geo = geo

    def add_branch(self, site_name, cur_state, new_branch):
        print("Branch added")
        return self.add_type(site_name, 'branch', cur_state, new_branch)

    def add_leaf(self, site_name, cur_state, new_leaf):
        print("Leaf added")
        return self.add_type(site_name, 'leaf', cur_state, new_leaf)

    def add_flower(self, site_name, cur_state, new_flower):
        print("Flower added")
        return self.add_type(site_name, 'flower', cur_state, new_flower)


def get_state(obj):
    return {
        "depth": obj["depth"],
        "pos": obj["pos"],
        "angle": obj["angle"],
        "width": obj["width"],
        "prev_branch": obj["prev_branch"],
        "features": None
    }


# Magic item, if you call a property, it will increment that property under
# the hood and return that id
class ObjectCounter(object):
    def __init__(self):
        self.counters = defaultdict(lambda: 0)
        pass

    # simple get attribute function return
    def __getitem__(self, name):
        count = self.counters[name]
        self.counters[name] += 1
        return "{}_{}".format(name, count)

    def __getattr__(self, name):
        return self.__getitem__(name)


#  ----------------------------------------------------------------------------
#  The program itself
def main(simTightness=0.02, boundsTightness=0.001,
         initial_width=0.35,
         widthDecay=0.975,
         min_width_percent=0.15,
         leafAspect=2.09859154929577,
         leafWidthMul=1.3,
         flowerRadMul=1,
         target_file=None):

    # for now, target is a global. Later, this can be sent in batches
    target = load_target(target_file)
    min_width = min_width_percent * initial_width
    start_viewport = Viewport(**{"xmin": -12, "xmax": 12, "ymin": -22, "ymax": 2})

    vis = Visdom(env='paint')
    vis.image(target.pt_img.data.numpy())

    # basic sim function, send in edge weight for closure
    similarity_fct = make_gradient_weight_similarity_fct(1.5)

    def target_factor(full_state, bbox, viewport, target):

        # render
        partial_render, viewport = render_update(full_state,
                                                 viewport,
                                                 print_render=True)

        # Similarity factor
        sim = normalized_similarity(partial_render, target, similarity_fct)

        # factor to add to observe log_pdf
        simf = makescore(sim, 1, simTightness)

        # Bounds factors
        extraX = (max(viewport.xmin - bbox.min.x, 0) +
                  max(bbox.max.x - viewport.xmax, 0)) / (viewport.xmax -
                                                         viewport.xmin)

        extraY = (max(viewport.ymin - bbox.min.y, 0) +
                  max(bbox.max.y - viewport.ymax, 0)) / (viewport.ymax -
                                                         viewport.ymin)

        boundsfx = makescore(extraX, 0, boundsTightness)
        boundsfy = makescore(extraY, 0, boundsTightness)

        # full factor = sum of parts
        f = simf + boundsfx + boundsfy

        # return the likelihood scoring relative to target, and our modified
        # viewpoint
        return f, viewport

    def model():

        # create a counting object
        oc = ObjectCounter()

        # Determine starting state by inverting viewport transform
        starting_world_pos = norm2world(target.start_pos, start_viewport)
        starting_dir = target.start_dir
        starting_ang = VTA(np.arctan2(starting_dir.y, starting_dir.x))

        # // These are separated like this so that we can have an initial local
        # //    state to feed to the _gaussian for the initial angle.
        init_state = get_state({
            "depth": 0,
            "pos": starting_world_pos,
            "angle": VTA(0),
            "width": VTA(initial_width),
            "prev_branch": None
        })

        start_state = get_state({
            "depth": init_state["depth"],
            "pos": init_state["pos"],
            "angle": (starting_ang, VTA(np.pi / 6)),
            "width": init_state["width"],
            "prev_branch": init_state["prev_branch"]
        })
        start_state["img_size"] = target.pt_img.data.shape[1:]
        start_state["viewport"] = start_viewport.clone()

        # hold our current procedural vine object
        proc_vines = ProceduralVine(start_state, BBOX2(), start_viewport,
                                    target, target_factor)

        leaf_opts = ['none', 'left', 'right']
        leaf_probs = norm_prob(VT([1, 1, 1]))

        # cluster around the procedural vines object
        def create_branch(cur_state):

            # calculate the width
            width = widthDecay * cur_state["width"]
            length = VTA(2)

            new_ang = cur_state["angle"] + proc_vines._gaussian(oc.angle,
                                                                VTA(0),
                                                                VTA(np.pi / 8))
            new_branch = {
                "start": cur_state["pos"],
                "angle": new_ang,
                "width": width,
                "end": polar2rect(length, new_ang).add(cur_state["pos"])
            }

            proc_vines.add_branch(oc.branch_obs, cur_state, new_branch)

            new_state = get_state({
                "depth": cur_state["depth"] + 1,
                "pos": new_branch["end"],
                "angle": new_branch["angle"],
                "width": new_branch["width"],
                "prev_branch": proc_vines.last_geo
            })

            # generate leaf (no futures yet) TODO: Add futures/pyro promises
            leaf_sopt = proc_vines._discrete3(oc.leaf, leaf_probs, vs=leaf_opts)

            if leaf_sopt != 'none':
                lwidth = VT([leafWidthMul * initial_width])
                llength = lwidth * leafAspect
                angmean = np.pi / 4 if leaf_sopt == 'left' else -np.pi / 4
                langle = new_branch["angle"] \
                    + proc_vines._gaussian(oc.leaf_angle, VTA(angmean), VTA(np.pi / 12)).data[0]

                lstart = new_branch["start"].clone().lerp(
                    new_branch["end"], 0.5)

                lend = polar2rect(llength, langle).add(lstart)
                lcenter = lstart.clone().add(lend).scalar_mult(0.5)
                # observe unique leaf ix, send in the state, along with the
                # leaf_obj
                # new leaf added!
                proc_vines.add_leaf(oc.leaf_obs,
                                    new_state,
                                    {
                                        "length": llength,
                                        "width": lwidth,
                                        "angle": langle,
                                        "center": lcenter
                                    })

            # Generate flower? TODO: Add futures/pyro flower
            if proc_vines._flip(oc.flower, VTA(0.5)):
                proc_vines.add_flower(oc.flower_obs, new_state, {
                    "center": new_branch["end"],
                    "radius": VTA(flowerRadMul * initial_width),
                    "angle": new_branch["angle"]
                })

            # Terminate? TODO: Add futures/pyro terminate
            terminate_prob = 0.25
            # do we terminate here and now?
            if proc_vines._flip(oc.terminate,
                                VTA(terminate_prob)):
                proc_vines.terminated = True
            else:
                # not done terminating yet!
                # Generate no further branches w/ prob 1/3
                # Generate one further branch w/ prob 1/3
                # Generate two further branches w/ prob 1/3
                # TODO: Add as future
                # pyro.future(lambda x: )
                if not proc_vines.terminated \
                        and new_state["width"].data[0] > min_width \
                        and proc_vines._flip(oc.reup_branch_one, VTA(0.66)):
                    # create another branch!
                    create_branch(new_state)

                    # TODO: Add as future pyro.future(lambda x: )
                    if not proc_vines.terminated and \
                            new_state["width"].data[0] > min_width and \
                            proc_vines._flip(oc.reup_branch_two, VTA(0.5)):

                        # if we fliipped yes, keep the good times going
                        create_branch(new_state)

        # create a branch of the start state
        create_branch(start_state)

        # once done, send back the vines object
        return proc_vines

    # now we define our inference alg
    # call our model plz
    model()

    # bb()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-cfg', "--config_file", type=str, required=True,
    # help='master configuration file for running the training experiments')
    parser.add_argument('-tgt', "--target_file", type=str, required=True,
                        help='target to match')

    parser.add_argument('-iw', "--initial_width", type=float,
                        help='width of lines to draw')

    args = parser.parse_args()

    # turn args into a list of kwargs sent to main
    main(**vars(args))
