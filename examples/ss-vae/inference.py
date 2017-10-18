
import math

class BaseInference(object):
    #Generalized to multiple KL_QPs
    def __init__(self, data, inference_techniques, conditions_apply_infer, do_classification=True):
        self.data = data
        self.inference_techniques = inference_techniques
        self.conditions_apply_infer = conditions_apply_infer
        self.num_losses = len(self.inference_techniques)
        assert self.num_losses >= 1, "need at least one loss"
        self.do_classification = do_classification


    def classify(self,xs):
        raise NotImplementedError("classify needs to be implemented")


    def run_inference_batches(self):
        epoch_losses = [0.]*self.num_losses
        batch_counts = [0] *self.num_losses
        self.data.reset_batch(training=True)
        while True:
            i, xs, ys = self.data.get_next_batch(training=True)
            if xs is None:
                break
            for loss_id in range(self.num_losses):
                if self.conditions_apply_infer[loss_id](i,xs,ys):
                    new_loss = self.inference_techniques[loss_id].step(i, xs, ys)
                    #print("new_loss: {}".format(new_loss))
                    assert not math.isnan(new_loss)
                    epoch_losses[loss_id] += new_loss
                    batch_counts[loss_id] += 1
        return epoch_losses, batch_counts

    def run(self, num_epochs=1000, acc_cutoff = 0.99,  *args, **kwargs):
        self.loss_training = []
        for i in range(num_epochs):
            epoch_losses, batch_counts = self.run_inference_batches()
            avg_epoch_losses = [0.]*self.num_losses
            for loss_id in range(self.num_losses):
                avg_epoch_losses[loss_id] = \
                    epoch_losses[loss_id]/(1.0*batch_counts[loss_id]*self.data.training_batch_size)

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
        self.data.reset_batch(training=training)
        while True:
            i,xs,ys = self.data.get_next_batch(training=training)
            if xs is None:
                break
            predictions.append(self.classify(xs))
        accuracy = self.data.prediction_accuracy(predictions, training=training)
        return accuracy