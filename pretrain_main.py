'''
prefetchDataset shapes: {image: (1, 512, 224, 224, 3), label: (1, 512, 10)}, types: {image: tf.float32, label: tf.float32}

'''

from os import name
import ml_collections
import tensorflow as tf
import tensorflow_addons as tfa
import math

from net.detection import Swin_Encoder
from net.pretrain_clsnet import Swin_CLS_Net
from pretrain_dataloader import get_data_from_tfds, get_dataset_info

import pretrain_config as training_config
import model_config


class Gradient_Accumulating_Model(tf.keras.Model):
    def __init__(self, inputs, outputs,n_gradients):
        super(Gradient_Accumulating_Model, self).__init__(inputs,outputs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

        
    def compile(self, optimizer, loss_fn, metrics):
        super(Gradient_Accumulating_Model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics_fn = metrics
    
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

            loss = self.loss_fn(y_true, y_pred)


        # Compute and Accumulate batch gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: print("accum step:", self.n_acum_step))

        # Update metrics (includes the metric that tracks the loss)
        self.metrics_fn.update_state(y_true, y_pred)

        # Return a dict mapping metric names to current value
        result = {"loss":loss, "acc":self.metrics_fn.result()}
        return result

if __name__ == "__main__":

    # tf.data.experimental.enable_debug_mode()
    # tf.config.experimental_run_functions_eagerly(True)
    tf.debugging.enable_check_numerics()

    gpus = tf.config.experimental.list_physical_devices('GPU')

    tf.config.experimental.set_visible_devices(gpus[0], "GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # initialize dataset
    dataset = "cifar10"
    # dataset = "food101"
    
    config = training_config.with_dataset(training_config.get_config(), dataset)
    ds_train_info = get_dataset_info(dataset, "train")
    ds_train_num_classes = ds_train_info['num_classes']
    ds_train_num_examples = ds_train_info["num_examples"]
    ds_train = get_data_from_tfds(config=config, mode='train')
    print("ds_train_info:",ds_train_info)

    ds_val_info = get_dataset_info(dataset, "test")
    # ds_val_info = get_dataset_info(dataset, "validation")
    ds_val_num_classes = ds_train_info['num_classes']
    ds_val_num_examples = ds_train_info["num_examples"]
    ds_val = get_data_from_tfds(config=config, mode='test')


    one_train_data = next(ds_train.as_numpy_iterator())[0]
    print("one_train_data.shape:", one_train_data["image"].shape) # vit_model_config 
    print(one_train_data["image"].shape[1:])




    # initialize model
    # vit_model_config = model_config.get_b32_config()
    swin_model_config = model_config.get_swin_config()

    swin_encoder = Swin_Encoder(config, \
            norm_layer=tf.keras.layers.LayerNormalization, **swin_model_config)

    print("ds_train_num_classes:",ds_train_num_classes)

    swin_cls_net = Swin_CLS_Net(num_classes=ds_train_num_classes, encoder = swin_encoder)


    # this init the model and avoid manipulate weight in graph(if using resnet)
    # trial_logit = vit_model(one_train_data["image"], train = True) # (512, 10) 
    # trial_logit = swin_model(one_train_data["image"]) # (512, 10) 

    # build model, expose this to show how to deal with dict as fit() input
    model_input = tf.keras.Input(shape=one_train_data["image"].shape[1:],name="image",dtype=tf.float32)

    # logit = vit_model(model_input)
    logit = swin_cls_net(model_input)

    prob = tf.keras.layers.Softmax(axis = -1, name = "label")(logit)

    # model = tf.keras.Model(inputs = [model_input],outputs = [prob], name = "ViT_model")

    
    GA_model = Gradient_Accumulating_Model(inputs = [model_input],outputs = [prob], n_gradients= 8)


    # import pdb
    # pdb.set_trace()


    '''
    the training config is for fine tune. I use my own config instead for training purpose.
    
    '''
    # my training config:
    steps_per_epoch = ds_train_num_examples//config.batch
    validation_steps = 3
    log_dir="./pretrain_tf_log/"
    total_steps = 100
    warmup_steps = 5
    base_lr = 1e-5

    # define callback 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./pretrain_weight/pretrain_classifier.ckpt',
        save_weights_only= True,
        verbose=1)

    callback_list = [tensorboard_callback,save_model_callback]


    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-2, decay_steps = 1000, decay_rate = 0.01, staircase=False, name=None)
    # lr_schedule = Cosine_Decay_with_Warm_up(base_lr, total_steps, warmup_steps)


    GA_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = base_lr), 
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False), # softmax included in model
        metrics=tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        )

    print(GA_model.summary())

    # import pdb
    # pdb.set_trace()


    # hist = GA_model.fit(ds_train,
    #             epochs=2000, 
    #             steps_per_epoch=steps_per_epoch,
    #             validation_data = ds_val,
    #             validation_steps=3,callbacks = callback_list).history
    
    hist = GA_model.fit(ds_train,
            epochs=1, 
            steps_per_epoch=1,
            validation_data = ds_val,
            validation_steps=3,callbacks = callback_list).history
    
    swin_encoder.save_weights(filepath="./pretrain_weight/swin_encoder.ckpt")




