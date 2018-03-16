import torch
import pyro

class AcquisitionFunction:
    """
    Base acquisition function.
    
    Keyword Arguments:
    :param GPRegression gp: Gaussasin Process Regression Module.
    """
    def __init__(self, gp):
        self.gp = gp

    def update(self):
        """
        Called once by the Bayesian Optimization Module every time it get the next point to quire.
        """
        pass
    
    @property
    def support_batch(self):
        return False
        
