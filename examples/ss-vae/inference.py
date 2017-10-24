
import math
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from pdb import set_trace as bb
from networks import USE_CUDA
from torch.autograd import Variable
import numpy as np

from torchvision.datasets import MNIST

from sys import stdout

# for a util object
# this is a print updater
def print_update_percent(ix, total, base_message):
  stdout.write("\r " + base_message + " {0:.0f}% ".format(100*(ix + 1.0)/total))
  stdout.flush()
  
# transformations for MNIST data
def fn_x_MNIST(x):
    xp = x*(1./255.)
    # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
    xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])
    xp = xp.view(-1,xp_1d_size)
    if USE_CUDA:
        xp = xp.cuda()
    return xp

def fn_y_MNIST(y):
    yp = torch.zeros(y.size(0),10)
    if USE_CUDA:
        yp = yp.cuda()
        y = y.cuda()
    yp = yp.scatter_(1, y.view(-1, 1), 1.0)
    return yp

def split_sup_unsup(X, y, sup_perc):
    # number of examples
    n = X.size()[0]

    # number of supervised examples
    sup_n = int(n*sup_perc/100.0)

    return X[range(sup_n)], y[range(sup_n)], X[range(sup_n,n)], y[range(sup_n,n)],

class MNIST_cached(MNIST):
    def __init__(self,train="sup",transform=fn_x_MNIST, target_transform=fn_y_MNIST,
                 sup_perc=5.0, *args,**kwargs):
        #init with no transforms
        super(MNIST_cached, self).__init__(train=train in ["sup","unsup"],*args,**kwargs)
        assert train in ["sup","unsup","test"], "invalid train values"

        if train in ["sup","unsup"]:
            if transform is not None:
                self.train_data = (transform(self.train_data.float()))
            if target_transform is not None:
                self.train_labels = (target_transform(self.train_labels))

            train_data_sup, train_labels_sup, train_data_unsup, train_labels_unsup = \
                split_sup_unsup(self.train_data,self.train_labels,sup_perc)
            if train == "sup":
                self.train_data, self.train_labels =  train_data_sup, train_labels_sup
            else:
                self.train_data = train_data_unsup
                self.train_labels = train_labels_unsup #TODO: make these Nones
            #self.train_data,self.train_labels = None, None
        else:
            if transform is not None:
                self.test_data = (transform(self.test_data.float()))
            if target_transform is not None:
                self.test_labels = (target_transform(self.test_labels))
            #for the whole dataset apply the transforms!
    def __getitem__(self, index):
        """
        Args:
            index (int): Index or slice object

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target


class BaseInference(object):
    #Generalized to multiple KL_QPs
    def __init__(self, dataset, batch_size, inference_techniques, is_supervised_loss,
                 do_classification=True, transform=fn_x_MNIST, target_transform=fn_y_MNIST,
                 sup_perc=5.0):
        self.dataset = dataset
        self.periodic_interval_batches = int(100/sup_perc)
        assert 100 % int(sup_perc) == 0, "only some perc values allowed n | 100"

        self.setup_data_loaders(batch_size, transform, target_transform, sup_perc=sup_perc)
        self.inference_techniques = inference_techniques
        self.is_supervised_loss = is_supervised_loss
        self.num_losses = len(self.inference_techniques)
        assert self.num_losses >= 1, "need at least one loss"
        self.do_classification = do_classification

    def setup_data_loaders(self, batch_size, transform, target_transform, root='./data',
                           download=True, sup_perc=5.0, **kwargs):
        self.batch_size = batch_size
        train_set_sup = self.dataset(root=root, train="sup", download=download,
                                 transform=transform, target_transform=target_transform,
                                 sup_perc=sup_perc)
        self.train_size_sup = len(train_set_sup)

        train_set_unsup = self.dataset(root=root, train="unsup", download=download,
                                 transform=transform, target_transform=target_transform,
                                 sup_perc=sup_perc)
        self.train_size_unsup = len(train_set_unsup)

        test_set = self.dataset(root=root, train="test",
                                transform=transform, target_transform=target_transform,
                                sup_perc=sup_perc)
        self.test_size = len(test_set)

        if 'num_workers' not in kwargs:
            kwargs = {'num_workers': 0, 'pin_memory': False}
        #TODO: cannot shuffle in the DataLoader during training -- the supervised and unsup parts may change
        self.train_loader_sup = DataLoader(train_set_sup,batch_size=batch_size, shuffle=True, **kwargs)
        self.train_loader_unsup = DataLoader(train_set_unsup, batch_size=batch_size, shuffle=True, **kwargs)

        num_batches = 0
        for _ in self.train_loader_sup:
            num_batches += 1
        for _ in self.train_loader_unsup:
            num_batches += 1
        self.batches_per_epoch = num_batches
        assert self.batches_per_epoch == len(self.train_loader_sup) + len(self.train_loader_unsup)

        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    def classify(self,xs):
        raise NotImplementedError("classify needs to be implemented")


    def run_inference_batches(self):
        epoch_losses = [0.]*self.num_losses
        batch_counts = [0] *self.num_losses
        sup_iter = iter(self.train_loader_sup)
        unsup_iter = iter(self.train_loader_unsup)
        for i in range(self.batches_per_epoch):
            is_supervised = i % self.periodic_interval_batches == 0
            if is_supervised:
                #supervised batch
                (xs, ys) = next(sup_iter)
            else:
                (xs, ys) = next(unsup_iter)
            xs,ys = Variable(xs), Variable(ys)
            print_update_percent(i,self.batches_per_epoch,"runnning this epoch")
            for loss_id in range(self.num_losses):
                if self.is_supervised_loss[loss_id] == is_supervised:
                    new_loss = self.inference_techniques[loss_id].step(is_supervised, xs, ys)
                    #assert not math.isnan(new_loss)
                    if math.isnan(new_loss):
                        print("Encountered nan loss, stopping training")
                        new_loss = 0.0
                        return None,None
                    epoch_losses[loss_id] += new_loss
                    batch_counts[loss_id] += 1
        return epoch_losses, batch_counts

    def run(self, num_epochs=1000, acc_cutoff = 0.99,  *args, **kwargs):
        self.loss_training = []
        for i in range(num_epochs):
            epoch_losses, batch_counts = self.run_inference_batches()
            if epoch_losses is None:
                break
            avg_epoch_losses = [0.]*self.num_losses
            for loss_id in range(self.num_losses):
                avg_epoch_losses[loss_id] = \
                    epoch_losses[loss_id]/(1.0*batch_counts[loss_id]*self.batch_size)

            self.loss_training.append(avg_epoch_losses)
            training_accuracy = self.get_accuracy(training=True)
            str_print = "{} epoch: avg losses {} training accuracy {}".\
                format(i," ".join(map(str,avg_epoch_losses)), training_accuracy)
            print(str_print)
            if self.do_classification and training_accuracy > acc_cutoff:
                break
        print "testing accuracy {}".format(self.get_accuracy(training=False))

    def get_accuracy(self,training=True):
        if not self.do_classification:
            return "classification disabled"
        predictions = []
        actuals = []

        def process(xs,ys):
            xs, ys = Variable(xs), Variable(ys)
            predictions.append(self.classify(xs))
            actuals.append(ys)

        if training:
            for (xs, ys) in self.train_loader_sup:
                process(xs,ys)
            for (xs, ys) in self.train_loader_unsup:
                process(xs, ys)
        else:
            for (xs,ys) in self.test_loader:
                process(xs,ys)
        accuracy = self.prediction_accuracy_computation(predictions, actuals)
        return accuracy

    def prediction_accuracy_computation(self, predictions, actuals):
        assert len(predictions) == len(actuals), "predictions,actuals size mismatch"
        accurate = 0
        for pred, act in zip(predictions,actuals):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate += (v.data[0] == 10)
        return accurate*1.0/(len(predictions)*self.batch_size)
