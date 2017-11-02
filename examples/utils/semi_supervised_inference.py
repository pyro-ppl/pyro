import torch
import sys
from torch.autograd import Variable
from utils.mnist_cached import setup_data_loaders


class SSInfer(object):
    """
        a wrapper around the inference algorithm that is generalized to handle
        multiple losses for semi-supervised data

        :param dataset: the data {x_i:images, y_i:labels} to run the inference on
        :param batch_size: size of a batch of data to use during training (and testing)
        :param losses: different losses to optimize
        :param is_supervised_loss: whether to use a loss while using supervised data or un-supervised data
        :param classify: how to classify an image (or a batch of images)
        :param sup_num: number of supervised (image, digit) pairs
        :param use_cuda: use GPU(s) to speed up training
        :param logger: a file object to write/log outputs to
    """
    def __init__(self, dataset, batch_size, losses,
                 classify, sup_num, use_cuda=True, logger=None):
        self.dataset = dataset
        self.sup_num = sup_num
        assert sup_num % batch_size == 0, "assuming simplicity of batching math"
        assert dataset.validation_size % batch_size == 0, "batch size should divide the " \
                                                          "number of validation examples"
        assert dataset.train_data_size % batch_size == 0, "batch size doesn't divide " \
                                                          "total number of training data examples"
        assert dataset.test_size % batch_size == 0, "batch size should divide the " \
                                                    "number of test examples"

        # how often would a supervised batch occur
        # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
        # until we have traversed through the appropriate number of batches
        self.periodic_interval_batches = int(dataset.train_data_size / (1.0 * sup_num))

        self.classify = classify
        self.use_cuda = use_cuda
        self.logger = logger
        self.batch_size = batch_size

        # setup data loaders
        self.train_loader_sup, self.train_loader_unsup, self.test_loader, self.validation_loader \
            = setup_data_loaders(dataset, use_cuda, batch_size, sup_num)

        # which losses to use for inference
        self.losses = losses

        self.num_losses = len(self.losses)
        assert self.num_losses >= 1, "inference needs at least one loss"

        # initializing local variables to maintain the best validation accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        self.best_valid_acc = 0.0
        self.corresponding_test_acc = None
        self.corresponding_state = None

    def print_and_log(self, msg):
        # print and log a message (if a logger is present)
        print(msg)
        sys.stdout.flush()
        if self.logger is not None:
            self.logger.write("{}\n".format(msg))
            self.logger.flush()

    def run_inference_for_epoch(self):
        """
             runs the inference algorithm for an epoch

             returns the values of all losses
        """
        # compute number of batches for an epoch
        sup_batches = len(self.train_loader_sup)
        unsup_batches = len(self.train_loader_unsup)
        batches_per_epoch = sup_batches + unsup_batches

        # initialize variables to store loss values
        epoch_losses_sup = [0.] * self.num_losses
        epoch_losses_unsup = [0.] * self.num_losses

        # setup the iterators for training data loaders
        sup_iter = iter(self.train_loader_sup)
        unsup_iter = iter(self.train_loader_unsup)

        # count the number of supervised batches seen in this epoch
        ctr_sup = 0
        for i in range(batches_per_epoch):

            # whether this batch is supervised or not
            is_supervised = (i % self.periodic_interval_batches == 1) and ctr_sup < sup_batches

            # extract the corresponding batch
            if is_supervised:
                (xs, ys) = next(sup_iter)
                ctr_sup += 1
            else:
                (xs, ys) = next(unsup_iter)
            xs, ys = Variable(xs), Variable(ys)

            # run the inference for each loss with supervised or un-supervised
            # data as arguments
            for loss_id in range(self.num_losses):
                if is_supervised:
                    new_loss = self.losses[loss_id].step(xs, ys)
                    epoch_losses_sup[loss_id] += new_loss
                else:
                    new_loss = self.losses[loss_id].step(xs)
                    epoch_losses_unsup[loss_id] += new_loss

        # return the values of all losses
        return epoch_losses_sup, epoch_losses_unsup

    def run(self, num_epochs=100):
        """
            run the inference for a given number of epochs
        :param num_epochs: number of epochs to run the inference for
        :return: None
        """
        # maintain training loss(es) across epochs
        self.loss_training = []

        for i in range(0, num_epochs):

            # get the losses
            epoch_losses_sup, epoch_losses_unsup = self.run_inference_for_epoch()

            # compute average epoch losses
            unsup_num = self.dataset.train_data_size - self.sup_num
            avg_epoch_losses_sup = map(lambda v: v / self.sup_num, epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)

            self.loss_training.append((avg_epoch_losses_sup, avg_epoch_losses_unsup))

            # store the loss and validation/testing accuracies in the logfile
            str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
            str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))

            str_print = "{} epoch: avg losses {}".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))
            validation_accuracy = self.get_accuracy(mode="valid")
            str_print += " validation accuracy {}".format(validation_accuracy)

            # this test accuracy is only for logging, this is not used
            # to make any decisions during training
            test_accuracy = self.get_accuracy(mode="test")
            str_print += " test accuracy {}".format(test_accuracy)

            # update the best validation accuracy and the corresponding
            # testing accuracy and the state of the parent module (including the networks)
            if self.best_valid_acc < validation_accuracy:
                self.best_valid_acc = validation_accuracy
                self.corresponding_test_acc = test_accuracy

            self.print_and_log(str_print)

        final_test_accuracy = self.get_accuracy(mode="test")
        self.print_and_log("best validation accuracy {} corresponding testing accuracy {} "
                           "last testing accuracy {}".format(self.best_valid_acc,
                                                             self.corresponding_test_acc,
                                                             final_test_accuracy)
                           )

    def get_accuracy(self, mode="train"):
        """
            compute the accuracy over the supervised training set or the testing set
        """
        predictions = []
        actuals = []

        assert mode in ["train", "test", "valid"], "invalid mode for get_accuracy"

        # use classify function to compute all predictions for each batch
        def process(xs, ys):
            xs, ys = Variable(xs), Variable(ys)
            predictions.append(self.classify(xs))
            actuals.append(ys)

        # use the appropriate data loader
        if mode == "train":
            for (xs, ys) in self.train_loader_sup:
                process(xs, ys)
        elif mode == "valid":
            for (xs, ys) in self.validation_loader:
                process(xs, ys)
        elif mode == "test":
            for (xs, ys) in self.test_loader:
                process(xs, ys)

        # compute the number of accurate predictions
        accurate_preds = 0
        for pred, act in zip(predictions, actuals):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds += (v.data[0] == 10)

        # calculate the accuracy between 0 and 1
        accuracy = (accurate_preds * 1.0) / (len(predictions) * self.batch_size)
        return accuracy
