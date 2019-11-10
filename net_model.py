from __future__ import print_function
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D, add, ZeroPadding2D, concatenate
from keras import Model
from keras.models import Sequential
import numpy as np
import hyper_parameters as paras
from custom_layer import Scale


def pool_layer(x, kernal_size, stride, name, idname):
    if name == 'max':
        pool = MaxPooling2D(pool_size=(kernal_size, kernal_size), strides=stride, padding='same', name=idname)(x)
    elif name == 'avg':
        pool = AveragePooling2D(pool_size=(kernal_size, kernal_size), strides=stride, padding='same', name=idname)(x)
    return pool

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    numbn = 0
    if name is not None:
        bn_name = name + '_bn%d' % numbn
        conv_name = name + '_conv%d' % numbn
    else:
        bn_name = None
        conv_name = None
    numbn += 1
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

def resnet_v1(inpt, nb_filter, kernel_size,idname, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet_v2(inpt, nb_filters, kernel_size, idname,strides=(1, 1), with_conv_shortcut=False):
    k1 = nb_filters
    k2 = nb_filters
    k3 = nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=kernel_size, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def transition_block(x, nb_filter, idname, compression=0.5, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + '_blk' + str(idname)
    relu_name_base = 'relu' + '_blk' + str(idname)
    pool_name_base = 'pool' + '_blk' + str(idname)

    x = BatchNormalization(epsilon=eps, name=conv_name_base+'_bn')(x)
    x = Scale(name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), kernel_size=(1, 1), strides=(1, 1), name=conv_name_base, padding='SAME', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(1, 1), padding='SAME', name=pool_name_base)(x)
    return x

def conv_block(x, branch, den_block_kernal, growth_rate, idname, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    conv_name_base = 'conv' + '_dense' + str(branch) + str(idname)
    relu_name_base = 'relu' + '_dense' + str(branch) + str(idname)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = growth_rate * 4
    x = BatchNormalization(name=conv_name_base)(x)
    x = Scale(name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, kernel_size=1, strides=(1, 1), padding='same',  name=conv_name_base+'_x1', bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(name=conv_name_base+'_x2_bn')(x)
    x = Scale(name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)

    x = Conv2D(growth_rate, kernel_size=(den_block_kernal, den_block_kernal),
               strides=(1, 1), padding='same', name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, den_block_kernal, num_filter, den_block_num, growth_rate, idname,
                dropout_rate=0.0, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''
    eps = 1.1e-5
    concat_feat = x
    for i in range(den_block_num):
        branch = i+1
        x = conv_block(concat_feat, branch, den_block_kernal, growth_rate, idname, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=3)
        if grow_nb_filters:
            num_filter += growth_rate
    return concat_feat, num_filter

def inference(x, py_layer, reuse, g_name):
    with g_name.as_default():
        model = Sequential()
        layers = []
        py_layer = py_layer
        conv_namenum = 0
        bn_nameum = 0
        af_namenum = 0
        fc_namenum = 0
        pool_namenum = 0
        den_blocknamenum = 0
        resnet_v1_blocknamenum = 0
        resnet_v2_blocknamenum = 0
        for item in py_layer:
            ### conv
            if len(item) == len(paras.conv):
                conv_namenum = conv_namenum + 1
                c_kernal = item[0]
                c_number = item[1]
                c_stride = item[2]
                c_padding = item[3]
                x = Conv2D(c_number, c_kernal, padding=c_padding, strides=c_stride,
                           name='conv_%d' % conv_namenum)(x)

            elif len(item) == 1:
                value = item[0]
                #### BN
                if (value == 'true') | (value == 'false'):
                    bn_nameum = bn_nameum + 1
                    if value == 'true':
                        x = BatchNormalization(axis=3, name='BN_%d' % bn_nameum)(x)
                    else:
                        x = x

                ### af
                elif (value == 'relu') | (value == 'softmax') | (value == 'softplus') \
                        | (value == 'sigmoid') | (value == 'tanh') | (value == 'none'):
                    af_namenum = af_namenum + 1
                    x = Activation(value, name='af_%d' % af_namenum)(x)

                ### fc
                else:
                    fc_num = item[0]
                    fc_namenum = fc_namenum + 1
                    if len(x.shape) == 4:
                        x = Flatten()(x)
                    x = Dense(fc_num, activation='softmax', kernel_initializer='he_normal',
                              name='fc_%d' % fc_namenum)(x)

            ### pool && densenet && blocks
            elif len(item) == 3:
                if item[2] in ['max', 'avg']:
                    p_kernal = item[0]
                    p_stride = item[1]
                    p_name = item[2]
                    pool_namenum = pool_namenum + 1
                    idname = 'pool_%d' % pool_namenum
                    x = pool_layer(x, p_kernal, p_stride, p_name, idname)

                ### resnet_blocks
                elif item[0] in ['a', 'b']:
                    flag_block = 'true'
                    b_num = item[1]
                    b_kernalsize = item[2]
                    if item[0] == 'a':
                        resnet_v1_blocknamenum = resnet_v1_blocknamenum + 1
                        idname = 'resnet_v1_blocknamenum_%d' % resnet_v1_blocknamenum
                        base = x.get_shape().as_list()[-1]
                        b_kernalnum_list = [base, 2 * base]
                        if base == 3:
                            flag_block = 'false'
                            b_kernalnum = base
                            print('残差块的输入通道数为3,不满足条件  需要重新变异：', b_kernalnum)
                        else:
                            b_kernalnum = np.random.choice(b_kernalnum_list)
                        for i in range(b_num):
                            x = resnet_v1(x, idname=idname, nb_filter=b_kernalnum, kernel_size=(b_kernalsize, b_kernalsize),
                                          with_conv_shortcut=True)
                    else:
                        b_kernalnum = x.get_shape().as_list()[-1]
                        resnet_v2_blocknamenum = resnet_v2_blocknamenum+1
                        idname = 'resnet_v2_blocknamenum_%d' % resnet_v2_blocknamenum
                        for i in range(b_num):
                            x = resnet_v2(x, nb_filters=b_kernalnum, with_conv_shortcut=True,
                                          kernel_size=b_kernalsize, idname=idname)

                ### densenet
                else:
                    growth_rate = item[0]
                    den_block_kernal = item[1]
                    den_block_num = item[2]
                    num_filter = x.get_shape().as_list()[-1]
                    den_blocknamenum = den_blocknamenum + 1
                    idname = 'dense_%d' % den_blocknamenum
                    x, num_filter = dense_block(x, den_block_kernal, num_filter, den_block_num, growth_rate,
                                                idname, dropout_rate=0.0, weight_decay=1e-4)
                    # Add transition_block
                    x = transition_block(x, num_filter, idname, dropout_rate=0.0, weight_decay=1e-4)
        return x

def test_graph(x, py_layer, g_name):
    with g_name.as_default():
        py_layer = py_layer
        input_tensor = Input(x)
        output = inference(input_tensor, py_layer, reuse=True, g_name=g_name)
        result = Model(inputs=input_tensor, outputs=output)
        return result