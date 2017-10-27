
import math
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from pdb import set_trace as bb
from torch.autograd import Variable
import numpy as np
from functools import reduce

from torchvision.datasets import MNIST


# transformations for MNIST data
def fn_x_MNIST(x, use_cuda):
    xp = x*(1./255.)
    # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
    xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])
    xp = xp.view(-1,xp_1d_size)
    if use_cuda:
        xp = xp.cuda()
    return xp

def fn_y_MNIST(y, use_cuda):
    yp = torch.zeros(y.size(0),10)
    if use_cuda:
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

class MNISTCached(MNIST):
    def __init__(self,train="sup", sup_perc=5.0, use_cuda=True, *args,**kwargs):
        super(MNISTCached, self).__init__(train=train in ["sup","unsup"],*args,**kwargs)

        #transformations on MNIST data (normalization and one-hot conversion for labels)
        transform = lambda x: fn_x_MNIST(x,use_cuda)
        target_transform = lambda y: fn_y_MNIST(y,use_cuda)

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
                self.train_labels = (torch.Tensor(train_labels_unsup.shape[0]).view(-1,1))*np.nan
                #    train_labels_unsup
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


class SSVAEInfer(object):
    #Generalized to handle multiple losses
    def __init__(self, dataset, batch_size, losses, is_supervised_loss,
                 classify, do_classification=True, sup_perc=5.0, checkpoint_fn=None,
                 start_epoch=0, check_nans=None, use_cuda=True, logger=None):
        self.dataset = dataset
        self.periodic_interval_batches = int(100/sup_perc)
        assert sup_perc < 1 or 100 % int(sup_perc) == 0, "only some perc values allowed n | 100"

        self.classify=classify
        self.use_cuda = use_cuda
        self.logger = logger
        self.setup_data_loaders(batch_size, sup_perc=sup_perc)
        self.losses = losses
        self.is_supervised_loss = is_supervised_loss
        self.num_losses = len(self.losses)
        assert self.num_losses >= 1, "need at least one loss"
        self.do_classification = do_classification
        self.checkpoint_fn = checkpoint_fn
        self.start_epoch = start_epoch
        self.best_train_test_acc = (0.0,None)
        self.check_nans = check_nans if check_nans is not None else (lambda: None)
        self.check_ctr = 0

    def print_and_log(self, msg):
        print(msg)
        if self.logger is not None:
            self.logger.write("{}\n".format(msg))

    def setup_data_loaders(self, batch_size, root='./data',
                           download=True, sup_perc=5.0, **kwargs):
        self.batch_size = batch_size
        train_set_sup = self.dataset(root=root, train="sup", download=download,
                                     sup_perc=sup_perc, use_cuda=self.use_cuda)

        self.train_size_sup = len(train_set_sup)
        train_set_unsup = self.dataset(root=root, train="unsup", download=download,
                                       sup_perc=sup_perc, use_cuda=self.use_cuda)
        self.train_size_unsup = len(train_set_unsup)

        test_set = self.dataset(root=root, train="test", sup_perc=sup_perc,
                                use_cuda=self.use_cuda)
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
            #print_update_percent(i,self.batches_per_epoch,"runnning this epoch")
            for loss_id in range(self.num_losses):
                if self.is_supervised_loss[loss_id] == is_supervised:
                    self.check_nans()
                    new_loss = self.losses[loss_id].step(is_supervised, xs, ys)
                    self.check_nans()
                    if math.isnan(new_loss):
                        self.print_and_log("Encountered nan loss, using 0.0 loss value")
                        bb()
                        new_loss = 0.0
                    epoch_losses[loss_id] += new_loss
                    batch_counts[loss_id] += 1
        return epoch_losses, batch_counts

    def run(self, num_epochs=1000, acc_cutoff = 0.99,  *args, **kwargs):
        self.loss_training = []
        for i in range(self.start_epoch,num_epochs):
            epoch_losses, batch_counts = self.run_inference_batches()
            if epoch_losses is None:
                break
            avg_epoch_losses = [0.]*self.num_losses
            for loss_id in range(self.num_losses):
                avg_epoch_losses[loss_id] = \
                    epoch_losses[loss_id]/(1.0*batch_counts[loss_id]*self.batch_size)

            self.loss_training.append(avg_epoch_losses)
            str_print = "{} epoch: avg losses {}".format(i," ".join(map(str,avg_epoch_losses)))
            if self.do_classification:
                training_accuracy = self.get_accuracy(training=True)
                str_print += " training accuracy {}".format(training_accuracy)


                # This test accuracy is not used for picking the best NN configs
                test_accuracy = self.get_accuracy(training=False)
                str_print += " test accuracy {}".format(test_accuracy)

                if self.best_train_test_acc[0] < training_accuracy:
                    self.best_train_test_acc = (training_accuracy,test_accuracy)

            self.print_and_log(str_print)

            if self.checkpoint_fn is not None:
                self.checkpoint_fn(i,training_accuracy,test_accuracy,"last")
                if self.best_train_test_acc[0] == training_accuracy:
                    self.checkpoint_fn(i, training_accuracy, test_accuracy, "best")
            if self.do_classification and training_accuracy > acc_cutoff:
                break
        self.print_and_log("best training accuracy {} corresponding testing accuracy {}".format(
            self.best_train_test_acc[0], self.best_train_test_acc[1]))

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
            #for (xs, ys) in self.train_loader_unsup:
            #    process(xs, ys)
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
