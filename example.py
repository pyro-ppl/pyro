import torch
import pyro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch.autograd import Variable

Sigma = Variable(1 * torch.ones(1,1))
mu = Variable(0 * torch.ones(1))

gaussian = pyro.distributions.MultivariateNormal(mu, Sigma)
in_ = Variable(torch.arange(-10, 10, 0.01).unsqueeze(1))
plt.plot(in_.data.numpy(), torch.exp(gaussian.batch_log_pdf(in_)).data.numpy())
plt.plot(in_.data.numpy(), 1/np.sqrt(np.pi)*torch.exp(-in_**2/2).data.numpy())
plt.show()

Sigma = Variable(10 * torch.randn(2, 2))
Sigma = Sigma @ Sigma.transpose(0, 1)
mu = Variable(-2.4 * torch.ones(2))
gaussian = pyro.distributions.MultivariateNormal(mu, Sigma)
in_ = Variable(torch.rand(2000, 2) * 2 - 1)

data = gaussian.sample(50000).data.numpy()
plt.hist2d(data[:, 0], data[:, 1], bins=100)
plt.show()

Sigma = Variable(10 * torch.randn(3, 3))
Sigma = Sigma @ Sigma.transpose(0, 1)
mu = Variable(-2.4 * torch.ones(3))
gaussian = pyro.distributions.MultivariateNormal(mu, Sigma)
in_ = Variable(torch.rand(2000, 2) * 2 - 1)

data = gaussian.sample(2000).data.numpy()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.show()