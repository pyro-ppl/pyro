# Author: Paul Szerlip Converted from Original JS code:
# https://github.com/probmods/webppl/blob/gh-pages-vinesDemoFreeSketch/demos/vines/wppl/vines.wppl
from pdb import set_trace as bb
import numpy as np
import torch
from torch.autograd import Variable as V
from torch import Tensor as T
import pyro.distributions as dist
import pyro
from collections import defaultdict
from render import Vector2, BBOX2, load_target

# ----------------------------------------------------------------------------
# Globals / constants


def VT(x):
    # create torch variable around tesnor with input X
    return V(T(x))


def norm2world(p, viewport):
    return Vector2(	viewport.xmin + p.x * (viewport.xmax - viewport.xmin),
                    viewport.ymin + p.y * (viewport.ymax - viewport.ymin))


def polar2rect(r, theta):
    return Vector2(r * torch.cos(theta), r * torch.sin(theta))


def union_bbox(*bboxes):
    raise NotImplementedError("Yet to combine bboxes")

# ----------------------------------------------------------------------------
# Factor encouraging similarity to target image


def bbox_branch(new_branch):
    raise NotImplementedError("Yet to create bbox for branch")


def bbox_leaf(new_leaf):
    raise NotImplementedError("Yet to create bbox for leaf")


def bbox_flower(new_flower):
    raise NotImplementedError("Yet to create bbox for flower")


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


def normalizedSimilarity(img, target):
    raise NotImplementedError("Yet to measure img similarity")


# // Render update


def renderUpdate(geo, viewport):
    # TODO: Render to 2D canvas via python TODO^2: Render to Unity
    raise NotImplementedError("Yet to render to canvas")
    # return img, viewport

    # utils.rendering.drawImgToRenderContext(globalStore.genImg);
    # utils.rendering.renderIncr(geo, viewport); globalStore.genImg =
    # utils.rendering.copyImgFromRenderContext();


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
        self.target_factor
        self.terminated = False

        # start angle is mu,sigma for sample -- replace is with a pyro sample
        initial_state['angle'] = self._gaussian(
            'start_angle', initial_state['angle'][0],
            initial_state['angle'][1])

        # set the state as an array
        self.geo_state = [initial_state]

    def _flip(self, name, p):
        return pyro.sample(name, dist.bernoulli, p)

    def _discrete3(self, name, ps):
        return pyro.sample(name, dist.categorical, ps)

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
        return self.add_type(site_name, 'branch', cur_state, new_branch)

    def add_leaf(self, site_name, cur_state, new_leaf):
        return self.add_type(site_name, 'leaf', cur_state, new_leaf)

    def add_flower(self, site_name, cur_state, new_flower):
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
         initialWidth=0.75,
         widthDecay=0.975,
         minWidthPercent=0.15,
         leafAspect=2.09859154929577,
         leafWidthMul=1.3,
         flowerRadMul=1,
         target_file=None):

    # for now, target is a global. Later, this can be sent in batches
    target = load_target(target_file)
    minWidth = minWidthPercent * initialWidth,
    start_viewport = {"xmin": -12, "xmax": 12, "ymin": -22, "ymax": 2}

    def target_factor(full_state, bbox, viewport, target):

        # render
        partial_render, viewport = renderUpdate(full_state, viewport)

        # Similarity factor
        sim = normalizedSimilarity(partial_render, target)

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
        starting_world_pos = norm2world(target["startPos"], start_viewport)
        starting_dir = target["startDir"]

        starting_ang = torch.atan2(starting_dir.y, starting_dir.x)

        # // These are separated like this so that we can have an initial local
        # //    state to feed to the _gaussian for the initial angle.
        init_state = get_state({
            "depth": 0,
            "pos": starting_world_pos,
            "angle": 0,
            "width": initialWidth,
            "prev_branch": None
        })

        start_state = get_state({
            "depth": init_state["depth"],
            "pos": init_state["pos"],
            "angle": (starting_ang, np.pi / 6),
            "width": init_state["width"],
            "prev_branch": init_state["prev_branch"]
        })

        # hold our current procedural vine object
        proc_vines = ProceduralVine(start_state, BBOX2(), start_viewport,
                                    target, target_factor)

        leaf_opts = ['none', 'left', 'right']
        leaf_probs = norm_prob(VT([1, 1, 1]))

        # cluster around the procedural vines object
        def create_branch(cur_state):

            # calculate the width
            width = VT([widthDecay * cur_state["width"]])
            length = VT([2])

            new_ang = cur_state["angle"] + proc_vines._gaussian(oc.angle,
                                                                0,
                                                                np.pi / 8)
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
            leaf_sopt = leaf_opts[proc_vines._discrete3(oc.leaf, leaf_probs)]

            if leaf_sopt != 'none':
                lwidth = VT([leafWidthMul * initialWidth])
                llength = lwidth * leafAspect
                angmean = np.pi / 4 if leaf_sopt == 'left' else -np.pi / 4
                langle = new_branch["angle"] \
                    + proc_vines._gaussian(oc.leaf_angle, angmean, np.pi / 12)

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
            if proc_vines._flip(oc.flower, VT([0.5])):
                proc_vines.add_flower(oc.flower_obs, new_state, {
                    "center": new_branch["end"],
                    "radius": flowerRadMul * initialWidth,
                    "angle": new_branch["angle"]
                })

            # Terminate? TODO: Add futures/pyro terminate
            terminate_prob = 0.5
            # do we terminate here and now?
            if proc_vines._flip(oc.terminate,
                                VT([terminate_prob])):
                proc_vines.terminated = True
            else:
                # not done terminating yet!
                # Generate no further branches w/ prob 1/3
                # Generate one further branch w/ prob 1/3
                # Generate two further branches w/ prob 1/3
                # TODO: Add as future
                # pyro.future(lambda x: )
                if not proc_vines.terminated \
                        and new_state["width"] > minWidth \
                        and proc_vines._flip(oc.reup_branch_one, VT([0.66])):
                    # create another branch!
                    create_branch(new_state)

                    # TODO: Add as future pyro.future(lambda x: )
                    if not proc_vines.terminated and \
                            new_state["width"] > minWidth and \
                            proc_vines._flip(oc.reup_branch_two, VT([0.5])):

                        # if we fliipped yes, keep the good times going
                        create_branch(new_state)

    # now we define our inference alg
    bb()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-cfg', "--config_file", type=str, required=True,
    # help='master configuration file for running the training experiments')
    args = parser.parse_args()

    # turn args into a list of kwargs sent to main
    main(**vars(args))
