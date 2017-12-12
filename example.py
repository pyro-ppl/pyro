import torch
import pyro
import matplotlib.pyplot as plt
from torch.autograd import Variable

Sigma = Variable(10 * torch.ones(1,1))
mu = Variable(-2.4 * torch.ones(1))

gaussian = pyro.distributions.MultivariateNormal(mu, Sigma)
in_ = Variable(torch.arange(-10,10,0.01).unsqueeze(1))
plt.plot(in_.data.numpy(), torch.exp(gaussian.batch_log_pdf(in_)).data.numpy())
plt.show()

Sigma = Variable(10 * torch.eye(2))
mu = Variable(-2.4 * torch.ones(2))
gaussian = pyro.distributions.MultivariateNormal(mu, Sigma)
in_ = Variable(torch.rand(2000, 2) * 2 - 1)
