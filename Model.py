import keras.backend as K
import math
from keras import layers
from keras import regularizers
from keras.layers import Input,Conv2D, GRU,Lambda, Dense, RepeatVector,Reshape,BatchNormalization
from keras.models import Model
import tensorflow as tf

def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y,0),20))(inputs)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
    x = clipped_relu(x)
    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2b')(x)
    x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)


    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x

#make models
def convolutional_model(input_shape, batch_size,num_frames = 160):
    
    def conv_res_block(input,filters,stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,kernel_size=5,strides=2,padding='same',kernel_initializer = 'glorot_uniform',
                   kernel_regularizer = regularizers.l2(l=0.00001),name=conv_name)(input)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        
        o = clipped_relu(o)
        
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o
    
    def cnn_component(input):
        x_ = conv_res_block(input, 64, stage=1)
        x_ = conv_res_block(x_, 128, stage=2)
        x_ = conv_res_block(x_, 256, stage=3)
        x_ = conv_res_block(x_, 512, stage=4)
        return x_

    inputs = Input(shape=input_shape)  # TODO the network should be definable without explicit batch shape
    x = cnn_component(inputs)  # .shape = (BATCH_SIZE , num_frames/16, 64/16, 512)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames/16), 2048)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  #shape = (BATCH_SIZE, 512)
    x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE , 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs, x, name='convolutional')

#     print(model.summary())
    return model