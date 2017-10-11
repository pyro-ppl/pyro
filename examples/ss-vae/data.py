
import torch
import pandas as pd
import torch.utils.data as tdata
import numpy as np
from torch.autograd import Variable
from pdb import set_trace as bb
from networks import USE_CUDA

# transformations for MNIST data
def fn_x_MNIST(x):
    xp = x *(1./255.)
    # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
    xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])
    return xp.view(-1,xp_1d_size)

def fn_y_MNIST(y):
    yp = torch.zeros(y.size(0),10)
    if USE_CUDA:
        yp=yp.cuda()
    yp = yp.scatter_(1, y.view(-1, 1), 1.0)
    return yp

class DatasetWrapper(object):
    def __init__(self, dataset, root="./data", download=True, y_transform=fn_y_MNIST,
                 loading_batch_size=128, x_transform=fn_x_MNIST,
                 training_batch_size=512, testing_batch_size=512, training_size=None, *args, **kwargs):

        self.train_set = dataset(root=root, train=True, download=download)
        self.test_set = dataset(root=root, train=False)

        self.training_size = training_size
        self.x_transform = x_transform
        self.y_transform = y_transform

        kwargs = {'num_workers': 1, 'pin_memory': True}
        self.train_loader = tdata.DataLoader(dataset=self.train_set,
                                             batch_size=loading_batch_size, shuffle=True, **kwargs)
        self.test_loader = tdata.DataLoader(dataset=self.test_set,
                                            batch_size=loading_batch_size, shuffle=False, **kwargs)
        print(" Training set size: {}, Testing set size: {}". format(
            len(self.train_loader.dataset.train_data), len(self.test_loader.dataset.test_data)))

        self.set_training_vars(training_batch_size, *args, **kwargs)
        self.set_testing_vars(testing_batch_size, *args, **kwargs)

        self.current_training_batch = 0
        self.current_testing_batch = 0

    def set_training_vars(self, training_batch_size, *args, **kwargs):
        train_data = self.train_loader.dataset.train_data[:self.training_size]
        train_labels = self.train_loader.dataset.train_labels[:self.training_size]

        if USE_CUDA:
            train_data = train_data.cuda()
        if USE_CUDA:
            train_labels = train_labels.cuda()

        self.var_x_train = Variable(self.x_transform(train_data.float()))
        self.var_y_train = Variable(self.y_transform(train_labels))
        self.train_data_size = len(train_data)

        assert len(self.var_x_train.size()) == 2, "x should be 1D"
        self.x_size = int(self.var_x_train.size()[1])
        assert len(self.var_y_train.size()) == 2, "y should be 1D"
        self.y_size = int(self.var_y_train.size()[1])

        self.training_batch_size = training_batch_size
        all_batches = np.arange(0, self.train_data_size, training_batch_size)

        if all_batches[-1] != self.train_data_size:
            all_batches = list(all_batches) + [self.train_data_size]
        self.train_batch_end_points = all_batches

    def set_testing_vars(self, testing_batch_size, *args, **kwargs):
        test_data = self.test_loader.dataset.test_data
        if USE_CUDA:
            test_data = test_data.cuda()
        test_labels = self.test_loader.dataset.test_labels
        if USE_CUDA:
            test_labels = test_labels.cuda()

        self.var_x_test = Variable(self.x_transform(test_data.float()))
        self.var_y_test = Variable(self.y_transform(test_labels))
        self.test_data_size = len(test_data)

        self.testing_batch_size = testing_batch_size
        all_batches = np.arange(0, self.test_data_size, testing_batch_size)

        if all_batches[-1] != self.test_data_size:
            all_batches = list(all_batches) + [self.test_data_size]
        self.test_batch_end_points = all_batches

    def get_batch(self,ix,training=True):
        try:
            if training:
                [start, end] = self.train_batch_end_points[ix:ix + 2]
                #return self.train_set[start:end]
                rx,ry = self.var_x_train[start:end], self.var_y_train[start:end]
            else:
                [start, end] = self.test_batch_end_points[ix:ix + 2]
                #return self.test_set[start:end]
                rx,ry =  self.var_x_test[start:end], self.var_y_test[start:end]
        except Exception as e:
            print("Exception :: {}".format(e))
            return None, None
        return rx,ry
        #return self.x_transform(rx), self.y_transform(ry)

    def reset_batch(self,training=True):
        if training:
            self.current_training_batch = 0
        else:
            self.current_testing_batch = 0

    def get_next_batch(self,training=True):
        if training:
            return self.get_next_training_batch()
        else:
            return self.get_next_testing_batch()

    def get_next_testing_batch(self):
        if self.current_testing_batch < len(self.test_batch_end_points)-1:
            (x_batch,y_batch) = self.get_batch(self.current_testing_batch,training=False)
            self.current_testing_batch += 1
            return self.current_testing_batch-1, x_batch, y_batch
        else:
            return  None, None, None
    def get_next_training_batch(self):
        if self.current_training_batch < len(self.train_batch_end_points)-1:
            (x_batch,y_batch) = self.get_batch(self.current_training_batch,training=True)
            self.current_training_batch += 1
            return self.current_training_batch-1, x_batch, y_batch
        else:
            return  None, None, None

    def prediction_accuracy(self, predictions, training=True):
        if training:
            end_points = self.train_batch_end_points
        else:
            end_points = self.test_batch_end_points
        assert len(predictions) == len(end_points)-1
        accurate = 0
        for ix in range(len(end_points)-1):
            (x_batch, y_batch) = self.get_batch(ix,training=training)
            for i in range(y_batch.size(0)):
                v = torch.sum(predictions[ix][i] == y_batch[i])
                accurate += (v.data[0] == 10)
        return accurate*1.0/end_points[-1]




class CustomDataset(tdata.Dataset):
    """
        Custom dataset.
        To support arbitrary CSV files
    """

    def __init__(self, root, train =True,
                 transform=None, target_transform=None, download=False):
        """
        Args:
            csv_{train/text}_file (string): Path to the csv file with data.
            train: use training part of
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv_file = "{}/{}.csv".format(root, "train" if train else "test")
        self.df = pd.read_csv(csv_file, keep_default_na=False, na_values=["NA"])
        self.df_x = self.df.loc(x_header)
        self.df_y = self.df.loc(y_header)

        #self.x_header = x_header
        #self.y_header = y_header
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df_x.iloc(idx)
        y = self.df_y.iloc(idx)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        sample = (x,y)

        return sample