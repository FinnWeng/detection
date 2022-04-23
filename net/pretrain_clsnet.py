import tensorflow as tf
import numpy as np



class Swin_CLS_Net(tf.keras.Model):
    def __init__(self, num_classes, encoder,norm_layer=tf.keras.layers.LayerNormalization):
        super(Swin_CLS_Net, self).__init__()

        self.num_classes = num_classes
        self.encoder = encoder


        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()

        self.head =tf.keras.layers.Dense(num_classes, name='head')

    
    def call(self, input):
        x = input

        # print("before encoder x.shape:", x.shape) # b, h,w,c
        route_1, route_2, x = self.encoder(x)
        # print("after encoder x.shape:", x.shape) # b, h,w,c

        b, h, w, c = x.shape
        # print("b,h,w,c:",b,h,w,c)
        x = tf.reshape(x, [-1, h*w, c])
        # print("x.shape:", x.shape) # b, h*w*c
        x = self.norm(x)
        x = self.avgpool(x)
        x = self.head(x)

        return x 