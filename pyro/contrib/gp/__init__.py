from __future__ import absolute_import, division, print_function


class InducingPoints(nn.Module):
    
    def __init__(self, Xu, name="inducing_points"):
        super(self, InducingPoints).__init__()
        self.inducing_points = Parameter(Xu)
        
    def forward(self):
        return self.inducing_points
