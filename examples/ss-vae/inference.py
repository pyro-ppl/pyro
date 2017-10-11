
from pyro.infer import KL_QP

class BaseInference(object):
    def __init__(self, data, inference_technique, do_classification=True, ):
        self.data = data
        self.inference_technique = inference_technique
        self.do_classification = do_classification
        self.aux_infer = None #for adding an auxiliary loss

    def classify(self,xs):
        raise NotImplementedError("classify needs to be implemented")

    def add_aux_loss(self, condition_apply_aux, aux_infer_technique):
        self.condition_apply_aux = condition_apply_aux
        self.aux_infer = aux_infer_technique

    def run_inference_batches(self):
        epoch_loss = 0.
        epoch_loss_aux = 0.
        self.data.reset_batch(training=True)
        while True:
            i, xs, ys = self.data.get_next_batch(training=True)
            if xs is None:
                break
            epoch_loss += self.inference_technique.step(i, xs, ys)
            if self.aux_infer is not None:
                if self.condition_apply_aux(i,xs,ys):
                    epoch_loss_aux += self.aux_infer.step(i,xs,ys)
        return epoch_loss, epoch_loss_aux

    def run(self, num_epochs=1000, batched_run=True, acc_cutoff = 0.99,  *args, **kwargs):
        self.loss_training = []
        if not batched_run:
            self.all_batches = [0,self.data.train_data_size]

        for i in range(num_epochs):
            epoch_loss, epoch_loss_aux = self.run_inference_batches()
            self.loss_training.append(epoch_loss / float(self.data.train_data_size))
            training_accuracy = self.get_accuracy(training=True)

            print("{} epoch: avg loss {} training accuracy {}".
                  format(i,self.loss_training[i], training_accuracy))
            if self.aux_infer is not None:
                print("{} epoch: avg aux loss {} ".format(i, epoch_loss_aux))
            if self.do_classification and training_accuracy > acc_cutoff:
                break
        print "testing accuracy {}".format(self.get_accuracy(training=False))

    def get_accuracy(self,training=True):
        if not self.do_classification:
            return "classification disabled"
        predictions = []
        self.data.reset_batch(training=training)
        while True:
            ix,xs,ys = self.data.get_next_batch(training=training)
            if xs is None:
                break
            predictions.append(self.classify(xs))
        accuracy = self.data.prediction_accuracy(predictions, training=training)
        return accuracy