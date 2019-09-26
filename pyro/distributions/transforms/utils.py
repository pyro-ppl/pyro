
def clamp_preserve_gradients(x, min, max):
    # This helper function clamps gradients but still passes through the gradient in clamped regions
    return x + (x.clamp(min, max) - x).detach()
