import tensorflow as tf
from net.vit import PatchEmbed, BasicLayer, PatchMerging
import numpy as np


class Detection_Net(tf.keras.Model):
    def __init__(self, config):
        super(Detection_Net, self).__init__()
        self.config = config
        self.flat1 = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(config.GRID_H*config.GRID_W)
        self.d2 = tf.keras.layers.Dense(config.GRID_H*config.GRID_W*config.BOX*(4 + 1 + config.num_of_labels))
        self.reshape1 = tf.keras.layers.Reshape([self.config.GRID_H, self.config.GRID_W, self.config.BOX, 4 + 1 + self.config.num_of_labels])
    
    def call(self, input):
        x = input
        x = self.flat1(x)
        x = self.d1(x)
        x = self.d2(x)
        x =self.reshape1(x)
        return x



def yolov2_basic_block(dim, kernel_size = 3):
    return [
        tf.keras.layers.Conv2D(dim, (kernel_size,kernel_size), strides=(1,1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1)
        ]

def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)

class YOLOV2_Net(tf.keras.Model):
    def __init__(self, config):
        super(YOLOV2_Net, self).__init__()
        self.config = config
        self.ly_list1 = []
        self.ly_list2 = []
        self.ly_list_skip = []
        self.ly_list3 = []

        # Layer 1
        self.ly_list1.extend(yolov2_basic_block(32))
        self.ly_list1.extend([tf.keras.layers.MaxPooling2D(pool_size=(2, 2))])

        # layers 2 
        self.ly_list1.extend(yolov2_basic_block(64))
        self.ly_list1.extend([tf.keras.layers.MaxPooling2D(pool_size=(2, 2))])

        # layers 3 
        self.ly_list1.extend(yolov2_basic_block(config.model_dim))

        # layers 4
        self.ly_list1.extend(yolov2_basic_block(64))

        # layers 5
        self.ly_list1.extend(yolov2_basic_block(config.model_dim))
        self.ly_list1.extend([tf.keras.layers.MaxPooling2D(pool_size=(2, 2))])

        # layer 6
        self.ly_list1.extend(yolov2_basic_block(config.model_dim*2))

        # layer 7
        self.ly_list1.extend(yolov2_basic_block(config.model_dim, 1))

        # layer 8
        self.ly_list1.extend(yolov2_basic_block(config.model_dim*2))
        self.ly_list1.extend([tf.keras.layers.MaxPooling2D(pool_size=(2, 2))])

        # Layer 9
        self.ly_list1.extend(yolov2_basic_block(config.model_dim*4))
        
        # layer 10
        self.ly_list1.extend(yolov2_basic_block(config.model_dim*2,1))

        # layer 11
        self.ly_list1.extend(yolov2_basic_block(config.model_dim*4))

        # layer 12
        self.ly_list1.extend(yolov2_basic_block(config.model_dim*2,1))

        # layer 13
        self.ly_list1.extend(yolov2_basic_block(config.model_dim*4))

        self.maxp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))


        # Layer 14
        self.ly_list2.extend(yolov2_basic_block(config.model_dim*8))

        # layer 15
        self.ly_list2.extend(yolov2_basic_block(config.model_dim*4,1))

        # layer 16
        self.ly_list2.extend(yolov2_basic_block(config.model_dim*8))

        # layer 17
        self.ly_list2.extend(yolov2_basic_block(config.model_dim*4,1))

        # layer 18
        self.ly_list2.extend(yolov2_basic_block(config.model_dim*8))

        # layer 19
        self.ly_list2.extend(yolov2_basic_block(config.model_dim*8))

        # layer 20
        self.ly_list2.extend(yolov2_basic_block(config.model_dim*8))

        # layer 21
        self.ly_list_skip.extend(yolov2_basic_block(64,1))
        self.ly_list_skip.extend([tf.keras.layers.Lambda(space_to_depth_x2)])

  

        # layer 22
        self.ly_list3.extend(yolov2_basic_block(config.model_dim*8))

        # layer 23
        self.ly_list3.extend([tf.keras.layers.Conv2D(self.config.BOX*(4+1+self.config.num_of_labels), (1,1), strides=(1,1), padding='same', use_bias=False)])

        self.reshape1 = tf.keras.layers.Reshape([self.config.GRID_H, self.config.GRID_W, self.config.BOX, 4 + 1 + self.config.num_of_labels])
    



    
    def call(self, input):
        x = input
        for ly in self.ly_list1:
            x = ly(x)
        skip_connection = x

        x = self.maxp1(x)

        for ly in self.ly_list2:
            x = ly(x)

        for ly in self.ly_list_skip:
            skip_connection = ly(skip_connection)

        x = tf.concat([skip_connection, x],axis = -1)

        for ly in self.ly_list3:
            x = ly(x)

        output = self.reshape1(x)
    
        return output




'''
YOLO V3
'''

def yolov3_basic_block(dim, kernel_size = 3, strides = 1, padding = "same", use_bn = True):

    if use_bn:
        return [
            tf.keras.layers.Conv2D(dim, (kernel_size,kernel_size), strides=(strides,strides), padding=padding, use_bias=False),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.BatchNormalization()
            
            ]
    else:
        return [
            tf.keras.layers.Conv2D(dim, (kernel_size,kernel_size), strides=(strides,strides), padding=padding, use_bias=True)
            ]


class YOLOV3_Res_Block(tf.keras.Model):
    def __init__(self, out_dim1, out_dim2):
        super(YOLOV3_Res_Block, self).__init__()

        self.ly_list1=[]
        self.ly_list1.extend(yolov3_basic_block(out_dim1, 1, 1))
        self.ly_list1.extend(yolov3_basic_block(out_dim2, 3, 1))

    def call(self, input):
        x = input
        for ly in self.ly_list1:
            x = ly(x)
        x = x + input

        return x


class DarkNet53_Net(tf.keras.Model):
    def __init__(self, config):
        super(DarkNet53_Net, self).__init__()
        self.config = config

        self.ly_list1 = []

        self.ly_list1.extend(yolov3_basic_block(32, 3, 1))
        self.ly_list1.extend(yolov3_basic_block(64, 3, 2, "SAME")) # downsample
        self.ly_list1.append(YOLOV3_Res_Block(32, 64))
        self.ly_list1.extend(yolov3_basic_block(config.model_dim, 3, 2, "SAME")) # downsample
        for i in range(2):
            self.ly_list1.append(YOLOV3_Res_Block( 64, config.model_dim))
        self.ly_list1.extend(yolov3_basic_block(config.model_dim*2, 3, 2, "SAME")) # downsample
        for i in range(8):
            self.ly_list1.append(YOLOV3_Res_Block( config.model_dim, config.model_dim*2))
        
        # output route_1

        self.ly_list2 = []
        self.ly_list2.extend(yolov3_basic_block(config.model_dim*4, 3, 2, "SAME")) # downsample
        for i in range(8):
            self.ly_list2.append(YOLOV3_Res_Block( config.model_dim*2, config.model_dim*4))

        # output route_2
        self.ly_list3 = []
        self.ly_list3.extend(yolov3_basic_block(config.model_dim*8, 3, 2, "SAME")) # downsample
        for i in range(4):
            self.ly_list3.append(YOLOV3_Res_Block( config.model_dim*4, config.model_dim*8))

        # output input_data
    
    def call(self, input):
        x = input
        # print("x:", x.shape)
        for ly in self.ly_list1:
            x = ly(x)
        # print("x:", x.shape)
        route_1 = x
        # print("route_1:",route_1.shape)
        for ly in self.ly_list2:
            x = ly(x)

        # print("x:", x.shape)
 
        route_2 = x
        # print("route_2:",route_2.shape)

        for ly in self.ly_list3:
            x = ly(x)
        # print("x:", x.shape)

        '''
        route_1: image_h/8
        route_2: image_h/16
        x(final):image_h/32
        '''

        
        return route_1, route_2, x


class Decoder_Net(tf.keras.Model):
    def __init__(self, config):
        super(Decoder_Net, self).__init__()
        self.config = config

        self.ly_list1 = []
        self.ly_list2 = []
        self.ly_list3 = []
        self.large_bbox_ly_list = []
        self.middle_bbox_ly_list = []
        self.small_bbox_ly_list = []

        self.ly_list1.extend(yolov3_basic_block(config.model_dim*4, 1, 1))
        self.ly_list1.extend(yolov3_basic_block(config.model_dim*8, 3, 1))
        self.ly_list1.extend(yolov3_basic_block(config.model_dim*4, 1, 1))
        self.ly_list1.extend(yolov3_basic_block(config.model_dim*8, 3, 1))
        self.ly_list1.extend(yolov3_basic_block(config.model_dim*4, 1, 1))

        self.large_bbox_ly_list.extend(yolov3_basic_block(config.model_dim*8, 3, 1))
        self.large_bbox_ly_list.extend(
            yolov3_basic_block(self.config.BOX*(4+1+self.config.num_of_labels), 1, 1, use_bn=False))
        
        self.reshape_large = tf.keras.layers.Reshape([self.config.IMAGE_H//(2**(3+2)) , self.config.IMAGE_W//(2**(3+2)) , self.config.BOX, 4 + 1 + self.config.num_of_labels])
    
        
        self.up_c_list1 = yolov3_basic_block(config.model_dim*2, 1, 1)
        self.up_1 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.ly_list2.extend(yolov3_basic_block(config.model_dim*2, 1, 1))
        self.ly_list2.extend(yolov3_basic_block(config.model_dim*4, 3, 1))
        self.ly_list2.extend(yolov3_basic_block(config.model_dim*2, 1, 1))
        self.ly_list2.extend(yolov3_basic_block(config.model_dim*4, 3, 1))
        self.ly_list2.extend(yolov3_basic_block(config.model_dim*2, 1, 1))

        self.middle_bbox_ly_list.extend(yolov3_basic_block(config.model_dim*4, 3, 1))
        self.middle_bbox_ly_list.extend(
            yolov3_basic_block(self.config.BOX*(4+1+self.config.num_of_labels), 1, 1, use_bn=False))
        
        self.reshape_middle = tf.keras.layers.Reshape([self.config.IMAGE_H//(2**(3+1)) , self.config.IMAGE_W//(2**(3+1)) , self.config.BOX, 4 + 1 + self.config.num_of_labels])


        self.up_c_list2 = yolov3_basic_block(config.model_dim, 1, 1)
        self.up_2 = tf.keras.layers.UpSampling2D(size=(2, 2))


        self.ly_list3.extend(yolov3_basic_block(config.model_dim, 1, 1))
        self.ly_list3.extend(yolov3_basic_block(config.model_dim*2, 3, 1))
        self.ly_list3.extend(yolov3_basic_block(config.model_dim, 1, 1))
        self.ly_list3.extend(yolov3_basic_block(config.model_dim*2, 3, 1))
        self.ly_list3.extend(yolov3_basic_block(config.model_dim, 1, 1))

        self.small_bbox_ly_list.extend(yolov3_basic_block(config.model_dim*2, 3, 1))
        self.small_bbox_ly_list.extend(
            yolov3_basic_block(self.config.BOX*(4+1+self.config.num_of_labels), 1, 1, use_bn=False))
        
        self.reshape_small = tf.keras.layers.Reshape([self.config.IMAGE_H//(2**(3+0)) , self.config.IMAGE_W//(2**(3+0)) , self.config.BOX, 4 + 1 + self.config.num_of_labels])


        


    
    def call(self, route_1, route_2, x):
        # print("decoder input x:", x.shape)

        for ly in self.ly_list1:
            x = ly(x)
        # print("after ly_list1 x:", x.shape)


        conv_large_bbox = x
        for ly in self.large_bbox_ly_list:
            conv_large_bbox = ly(conv_large_bbox)
        conv_large_bbox = self.reshape_large(conv_large_bbox)
        # print("conv_large_bbox:",conv_large_bbox.shape)
        
        

        # updampling
        for ly in self.up_c_list1:
            x = ly(x)
        # print("before_up_1_x:", x.shape)
        x =  self.up_1(x)
        # print("up_1_x:", x.shape)
        # print("route_2:", route_2.shape)
        x = tf.concat([x, route_2], axis=-1)


        for ly in self.ly_list2:
            x = ly(x)
        
        conv_middle_bbox = x
        for ly in self.middle_bbox_ly_list:
            conv_middle_bbox = ly(conv_middle_bbox)
        
        conv_middle_bbox = self.reshape_middle(conv_middle_bbox)
        # print("conv_middle_bbox:",conv_middle_bbox.shape)
        
        # updampling
        for ly in self.up_c_list2:
            x = ly(x)
        x =  self.up_2(x)
        # print("up_2_x:", x.shape)
        # print("route_1:", route_1.shape)
        x = tf.concat([x, route_1], axis=-1)



        for ly in self.ly_list3:
            x = ly(x)
        
        conv_small_bbox = x
        for ly in self.small_bbox_ly_list:
            conv_small_bbox = ly(conv_small_bbox)
        # print("conv_small_bbox:",conv_small_bbox.shape)

        conv_small_bbox = self.reshape_small(conv_small_bbox)
        
        return conv_small_bbox, conv_middle_bbox, conv_large_bbox
        
        

class YOLOV3_Net(tf.keras.Model):
    def __init__(self, config):
        super(YOLOV3_Net, self).__init__()
        self.config = config
        self.encoder = DarkNet53_Net(config)
        self.decoder = Decoder_Net(config)

        self.reshape1 = tf.keras.layers.Reshape([self.config.GRID_H, self.config.GRID_W, self.config.BOX, 4 + 1 + self.config.num_of_labels])
    


    
    def call(self, input):
        x = input
        route_1, route_2, x = self.encoder(x)
        conv_small_bbox, conv_middle_bbox, conv_large_bbox = self.decoder(route_1, route_2, x)

        return conv_small_bbox, conv_middle_bbox, conv_large_bbox

    


'''
swin yolov3
'''

class Swin_Encoder(tf.keras.Model):
    def __init__(self,config, model_name='swin_tiny_patch4_window7_224', 
                 img_size=(224, 224), patch_size=(4, 4), in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=tf.keras.layers.LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs, ):
        super(Swin_Encoder, self).__init__()
        self.config = config
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                    shape=(
                                                        1, num_patches, embed_dim),
                                                    initializer=tf.initializers.Zeros())

        self.pos_drop = tf.keras.layers.Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers
        self.basic_layers = [BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)]
        
        

        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()

    def root_shape(self, tensor):
        tensor = tf.reshape(tensor,[-1,int(tensor.shape[1]**0.5),int(tensor.shape[1]**0.5),tensor.shape[2]])
        return tensor

    def call(self, input):
        x = input
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        output_list = []
        for i_layer in range(self.num_layers):
            # print("x.shape:", x.shape)
            x = self.basic_layers[i_layer](x)
            if i_layer < self.num_layers - 1:
                output_list.append(x)
            else:
                print("skip last!!")
                # print("last x.shape:", x.shape)


        route_1 = output_list[-3]
        route_2 = output_list[-2]

        route_2 = self.root_shape(route_2)
        route_1 = self.root_shape(route_1)
        x = self.root_shape(x)
        '''
        route_1: image_h/8
        route_2: image_h/16
        x(final):image_h/32
        '''

        
        return route_1, route_2, x



class Swin_YOLOV3_Net(tf.keras.Model):
    def __init__(self, config, swin_model_config):
        super(Swin_YOLOV3_Net, self).__init__()
        self.config = config
        self.encoder = Swin_Encoder(config, num_classes=config.num_of_labels, \
            norm_layer=tf.keras.layers.LayerNormalization,**swin_model_config )
        self.decoder = Decoder_Net(config)

        self.reshape1 = tf.keras.layers.Reshape([self.config.GRID_H, self.config.GRID_W, self.config.BOX, 4 + 1 + self.config.num_of_labels])
    


    
    def call(self, input):
        x = input
        route_1, route_2, x = self.encoder(x)
        conv_small_bbox, conv_middle_bbox, conv_large_bbox = self.decoder(route_1, route_2, x)

        return conv_small_bbox, conv_middle_bbox, conv_large_bbox
