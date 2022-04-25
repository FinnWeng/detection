import tensorflow as tf


class Gradient_Accumulating_Model(tf.keras.Model):
    # def __init__(self, inputs, outputs,n_gradients, metrics_list):
    def __init__(self, inputs, outputs,n_gradients):

        super(Gradient_Accumulating_Model, self).__init__(inputs,outputs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]
        # self.loss_tracker = metrics_list[0]
        # self.acc_metric = metrics_list[1]

        
    # def compile(self, optimizer, loss_fn):
    #     super(Gradient_Accumulating_Model, self).compile()
    #     self.optimizer = optimizer
    #     self.loss_fn = loss_fn
    #     # self.metrics_fn = metrics

    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [self.loss_tracker, self.acc_metric]
    
    def apply_accu_gradients(self):
        '''
        For solution of https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras,
        there's a obvious mistake that it simply assign_add the gradients and will make it self.n_gradients times greater(maximum case).
        So I feel it should be mean of gradients, not sum of gradient.
        '''
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(self.gradient_accumulation[i]/tf.cast(self.n_gradients, tf.float32))

        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))
        print("accumulated Gradient applied!")

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


    def train_step(self, data):
        '''
        {"image": im}, {"label": label}
        '''
        self.n_acum_step.assign_add(1)

        x, y = data
        x_batch = x["image"]
        y_true = y["label"]

        with tf.GradientTape() as tape:
            y_pred = self(x_batch)  # Forward pass

            # loss = self.loss_fn(y_true, y_pred)
            loss = self.compiled_loss(y_true, y_pred)


        # Compute and Accumulate batch gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: print("accum step:", self.n_acum_step))

        # Update metrics (includes the metric that tracks the loss)
        # self.metrics_fn.update_state(y_true, y_pred)ZZ

        # Return a dict mapping metric names to current value
        # result = {"loss":loss, "acc":self.metrics_fn.result()}
        # self.loss_tracker.update_state(loss)
        # self.acc_metric.update_state(y_true, y_pred)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=None)
        # return {"loss": self.loss_tracker.result(), "acc": self.acc_metric.result()}
        return {m.name: m.result() for m in self.metrics}

