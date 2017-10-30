
import torch
from torch.autograd import Variable
from data_cached import setup_data_loaders

class SSVAEInfer(object):
    """
        a wrapper around the inference algorithm that is generalized to handle
        multiple losses for semi-supervised data

        :param dataset: the data {x_i:images, y_i:labels} to run the inference on
        :param batch_size: size of a batch of data to use during training (and testing)
        :param losses: different losses to optimize
        :param is_supervised_loss: whether to use a loss while using supervised data or un-supervised data
        :param classify: how to classify an image (or a batch of images)
        :param sup_perc: % of data that is supervised
        :param use_cuda: use GPU(s) to speed up training
        :param logger: a file object to write/log outputs to
    """
    def __init__(self, dataset, batch_size, losses, is_supervised_loss,
                 classify, sup_perc=5.0, use_cuda=True, logger=None):
        self.dataset = dataset

        # how often would a supervised batch occur
        # e.g. if sup_perc is 5.0, we would have every 20th (=100/5) batch supervised
        self.periodic_interval_batches = int(100/sup_perc)
        assert sup_perc < 1 or 100 % int(sup_perc) == 0, "only some percentage values " \
                                                         "allowed for simple batching"

        self.classify=classify
        self.use_cuda = use_cuda
        self.logger = logger
        self.batch_size = batch_size

        # setup data loaders
        self.train_loader_sup, self.train_loader_unsup, self.test_loader \
            = setup_data_loaders(dataset, use_cuda, batch_size, sup_perc)

        # which losses to use and when
        self.losses = losses
        self.is_supervised_loss = is_supervised_loss

        self.num_losses = len(self.losses)
        assert self.num_losses >= 1, "inference needs at least one loss"

        # initializing local variables to maintain the best training accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set
        self.best_train_acc = 0.0
        self.corresponding_test_acc = None

    def print_and_log(self, msg):
        # print and log a message (if a logger is present)
        print(msg)
        if self.logger is not None:
            self.logger.write("{}\n".format(msg))


    def run_inference_for_epoch(self):
        """
             runs the inference algorithm for an epoch

             returns the values of all losses and the corresponding number of
             batches each loss was used for (this is used for average loss computation)

        """
        # compute number of batches for an epoch
        batches_per_epoch = len(self.train_loader_sup) + len(self.train_loader_unsup)

        # initialize variables to store loss values and batch counts
        epoch_losses = [0.]*self.num_losses
        batch_counts = [0] *self.num_losses

        # setup the iterators for training data loaders
        sup_iter = iter(self.train_loader_sup)
        unsup_iter = iter(self.train_loader_unsup)
        for i in range(batches_per_epoch):

            # whether this batch is supervised or not
            is_supervised = i % self.periodic_interval_batches == 0

            # extract the corresponding batch
            if is_supervised:
                (xs, ys) = next(sup_iter)
            else:
                (xs, ys) = next(unsup_iter)
            xs,ys = Variable(xs), Variable(ys)

            # run the inference for each loss with supervised or un-supervised
            # data as arguments
            for loss_id in range(self.num_losses):
                if self.is_supervised_loss[loss_id] == is_supervised:
                    if is_supervised:
                        new_loss = self.losses[loss_id].step(xs, ys)
                    else:
                        new_loss = self.losses[loss_id].step(xs)
                    epoch_losses[loss_id] += new_loss
                    batch_counts[loss_id] += 1
        # return the values of all losses and the corresponding number
        # of batches each loss was used for
        return epoch_losses, batch_counts

    def run(self, num_epochs=100):
        """
            run the inference for a given number of epochs
        :param num_epochs: number of epochs to run the inference for
        :return: None
        """
        # maintain training loss(es) across epochs
        self.loss_training = []

        for i in range(0,num_epochs):

            # get the losses and batch counts for current epoch
            epoch_losses, batch_counts = self.run_inference_for_epoch()

            # compute average epoch loss based on number of batches
            # each loss was used for over the full data
            avg_epoch_losses = [0.]*self.num_losses
            for loss_id in range(self.num_losses):
                avg_epoch_losses[loss_id] = \
                    epoch_losses[loss_id]/(1.0*batch_counts[loss_id]*self.batch_size)

            self.loss_training.append(avg_epoch_losses)

            # log the loss and training/testing accuracies
            str_print = "{} epoch: avg losses {}".format(i," ".join(map(str,avg_epoch_losses)))
            training_accuracy = self.get_accuracy(training=True)
            str_print += " training accuracy {}".format(training_accuracy)


            # This test accuracy is only for logging, this is not used
            # to make any decisions during training
            test_accuracy = self.get_accuracy(training=False)
            str_print += " test accuracy {}".format(test_accuracy)

            if self.best_train_acc < training_accuracy:
                self.best_train_acc = training_accuracy
                self.corresponding_test_acc = test_accuracy

            self.print_and_log(str_print)

        self.print_and_log("best training accuracy {} corresponding testing accuracy {} "
                           "last testing accuracy {}".format(self.best_train_acc,
                                                        self.corresponding_test_acc,
                                                        self.get_accuracy(training=False)
                                                        )
                           )

    def get_accuracy(self,training=True):
        """
            compute the accuracy over the supervised training set or the testing set
        """
        predictions = []
        actuals = []

        # use classify function to compute all predictions for each batch
        def process(xs,ys):
            xs, ys = Variable(xs), Variable(ys)
            predictions.append(self.classify(xs))
            actuals.append(ys)

        # use the appropriate data loader
        if training:
            for (xs, ys) in self.train_loader_sup:
                process(xs,ys)
        else:
            for (xs,ys) in self.test_loader:
                process(xs,ys)

        # compute the number of accurate predictions
        accurate_preds = 0
        for pred, act in zip(predictions, actuals):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds += (v.data[0] == 10)

        # calculate the accuracy between 0 and 1
        accuracy =  accurate_preds * 1.0 / (len(predictions) * self.batch_size)
        return accuracy


