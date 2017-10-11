

import torch
import torch.nn as nn
# networks

# For MNIST: values in tensor_sizes
# output_size = 10
# input_size = 28 * 28  #images 28 * 28 = 784 pixels
# latent_size = 20
USE_CUDA = True

class NNWithSizes(nn.Module):
    def __init__(self,tensor_sizes):
        super(NNWithSizes, self).__init__()
        self.input_size = tensor_sizes["input_size"]
        self.output_size = tensor_sizes["output_size"]
        self.hidden_sizes = tensor_sizes["hidden_sizes"]
        self.latent_size = tensor_sizes["latent_size"]


class Encoder_c(NNWithSizes):
    def __init__(self,tensor_sizes):
        super(Encoder_c, self).__init__(tensor_sizes)
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.fc21 = nn.Linear(self.hidden_sizes[0], self.output_size)
        if USE_CUDA:
            self.fc21 = nn.DataParallel(self.fc21)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.softmax(self.fc21(h1))

class Encoder_o(NNWithSizes):
    def __init__(self,tensor_sizes):
        super(Encoder_o, self).__init__(tensor_sizes)
        self.fc1 = nn.Linear(self.input_size +  self.output_size,
                             self.hidden_sizes[0])
        self.fc21 = nn.Linear(self.hidden_sizes[0],self.latent_size)
        if USE_CUDA:
            self.fc21 = nn.DataParallel(self.fc21)
        self.fc22 = nn.Linear(self.hidden_sizes[0], self.latent_size)
        if USE_CUDA:
            self.fc22 = nn.DataParallel(self.fc22)
        self.relu = nn.ReLU()

    def forward(self, x, cll):
        input_vec = torch.cat((x, cll), 1)
        h1 = self.relu(self.fc1(input_vec))
        return self.fc21(h1), torch.exp(self.fc22(h1))

class Decoder(NNWithSizes):
    def __init__(self,tensor_sizes):
        super(Decoder, self).__init__(tensor_sizes)
        self.fc3 = nn.Linear(self.latent_size +  self.output_size,
                             self.hidden_sizes[0])
        self.fc4 = nn.Linear(self.hidden_sizes[0], self.input_size)
        if USE_CUDA:
            self.fc4 = nn.DataParallel(self.fc4)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z, cll):
        input_vec = torch.cat((z, cll), 1)
        h3 = self.relu(self.fc3(input_vec))
        rv = self.sigmoid(self.fc4(h3))
        return rv


class Decoder_M1(NNWithSizes):
    def __init__(self,tensor_sizes):
        super(Decoder_M1, self).__init__(tensor_sizes)
        self.fc3 = nn.Linear(self.latent_size,
                             self.hidden_sizes[0])
        self.fc4 = nn.Linear(self.hidden_sizes[0], self.input_size)
        if USE_CUDA:
            self.fc4 = nn.DataParallel(self.fc4)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        rv = self.sigmoid(self.fc4(h3))
        return rv

class Encoder_M1_o(NNWithSizes):
    def __init__(self,tensor_sizes):
        super(Encoder_M1_o, self).__init__(tensor_sizes)
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.fc21 = nn.Linear(self.hidden_sizes[0],self.latent_size)
        self.fc22 = nn.Linear(self.hidden_sizes[0], self.latent_size)
        if USE_CUDA:
            self.fc21 = nn.DataParallel(self.fc21)
        if USE_CUDA:
            self.fc22 = nn.DataParallel(self.fc22)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), torch.exp(self.fc22(h1))