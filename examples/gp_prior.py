import torch
import pyro
import matplotlib.pyplot as plt
from torch.autograd import Variable

kernel = lambda x, y: 1 / (1 + (x - y) ** 2)

x_grid = torch.arange(-5, 5, 0.01)
covar = kernel(x_grid.view(1, 1001), x_grid.view(1001, 1))
covar -= torch.eye(covar.size()[0]) * min(0, torch.min(torch.eig(covar)[0][:, 0])-1e-6)

if __name__ == '__main__':
    gp = pyro.distributions.MultivariateNormal(Variable(torch.zeros(1001)), Variable(covar))
    plt.plot(x_grid.numpy(), gp.sample(5).data.numpy().transpose())
    plt.show()
