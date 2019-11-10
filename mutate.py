# -*- encoding: utf8 -*-
from numpy import random
import numpy as np
import hyper_parameters as params

def random_pick(some_list, probabilities, probabilitiesadd):
    global item
    x = random.uniform(0, probabilitiesadd)
    cumulative_probability = 0.0
    i = -1
    for item, item_probability in zip(some_list, probabilities):

        cumulative_probability += item_probability
        i = i+1
        if x < cumulative_probability:
            break
    return item, i

def convinit():
    p_c, p_conv_kernalsize, p_conv_num, p_conv_strides, p_conv_padding = [], [], [], [], []
    for i in range(len(params.conv)):
        p_c.append(1/len(params.conv))

    for i in range(len(params.Choice_kernal)):
        p_conv_kernalsize.append(1/len(params.Choice_kernal))

    for i in range(len(params.Choice_channel)):
        p_conv_num.append(1/len(params.Choice_channel))

    for i in range(len(params.Choice_stride)):
        p_conv_strides.append(1/len(params.Choice_stride))

    for i in range(len(params.Choice_padding)):
        p_conv_padding.append(1/len(params.Choice_padding))

    p_c_layers = [p_conv_kernalsize, p_conv_num, p_conv_strides, p_conv_padding]
    return p_c, p_c_layers

def poolinit():
    p_pool, p_pool_kernal, p_pool_stride, p_pool_name = [], [], [], []
    for i in range(len(params.pool)):
        p_pool.append(1/len(params.pool))

    for i in range(len(params.Choice_pool_kernal)):
        p_pool_kernal.append(1/len(params.Choice_pool_kernal))

    for i in range(len(params.Choice_stride)):
        p_pool_stride.append(1/len(params.Choice_stride))

    for i in range(len(params.Choice_poolname)):
        p_pool_name.append(1/len(params.Choice_poolname))

    p_pool_layers = [p_pool_kernal, p_pool_stride, p_pool_name]
    return p_pool, p_pool_layers

def bninit():
    p_bn, p_isbn = [], []
    for i in range(len(params.bn)):
        p_bn.append(1/len(params.bn))

    for i in range(len(params.Choice_bn)):
        p_isbn.append(1/len(params.Choice_bn))

    p_bn_layers = [p_isbn]
    return p_bn, p_bn_layers

def blocksinit():
    p_blocks, p_resblocks_type, p_blocks_num, p_blocks_kernal = [], [], [], []
    for i in range(len(params.resnet_blocks)):
        p_blocks.append(1/len(params.resnet_blocks))

    for i in range(len(params.Chioce_resblocks_type)):
        p_resblocks_type.append(1/len(params.Chioce_resblocks_type))

    for i in range(len(params.Choice_block_num)):
        p_blocks_num.append(1/len(params.Choice_block_num))

    for i in range(len(params.Choice_kernal)):
        p_blocks_kernal.append(1/len(params.Choice_kernal))

    p_blocks_layers = [p_resblocks_type, p_blocks_num, p_blocks_kernal]
    return p_blocks, p_blocks_layers

def denseinit():
    p_dense, p_dense_k, p_dense_kernal, p_dense_num = [], [], [], []
    for i in range(len(params.dense_blocks)):
        p_dense.append(1/len(params.dense_blocks))

    for i in range(len(params.Chioce_dense_k)):
        p_dense_k.append(1/len(params.Chioce_dense_k))

    for i in range(len(params.Choice_kernal)):
        p_dense_kernal.append(1/len(params.Choice_kernal))

    for i in range(len(params.Chioce_dense_num)):
        p_dense_num.append(1/len(params.Chioce_dense_num))

    p_dense_layers = [p_dense_k, p_dense_kernal, p_dense_num]
    return p_dense, p_dense_layers

def afinit():
    p_af, p_afname = [], []
    for i in range(len(params.af)):
        p_af.append(1/len(params.af))

    for i in range(len(params.Choice_af)):
        p_afname.append(1/len(params.Choice_af))

    p_af_layers = [p_afname]
    return p_af, p_af_layers

def fcinit():
    p_fc, p_fcnum = [], []
    for i in range(len(params.fc)):
        p_fc.append(1 / len(params.fc))

    for i in range(len(params.Choice_fc)):
        p_fcnum.append(1 / len(params.Choice_fc))

    p_fc_layers = [p_fcnum]
    return p_fc, p_fc_layers

def geneinit():
    p_gene = {}
    for i in params.gene:
        p_gene[str(i)] = 1/len(params.gene)
    return p_gene

def mutate_opinit():
    p_operate = []
    for i in range(params.OPERATENUM):
        p_operate.append(1/params.OPERATENUM)
    return p_operate

def initmuteta(py_layers):
    p_gene_para, p_layers = [], []
    for item in py_layers:
        if len(item) == len(params.conv):
            p_c, p_c_layers = convinit()
            p_gene_para.append(p_c)
            p_layers.append(p_c_layers)
        elif len(item) == 3:
            if item[2] in params.Choice_poolname:
                p_pool, p_pool_layers = poolinit()
                p_gene_para.append(p_pool)
                p_layers.append(p_pool_layers)
            elif item[0] in params.Chioce_resblocks_type:
                p_blocks, p_blocks_layers = blocksinit()
                p_gene_para.append(p_blocks)
                p_layers.append(p_blocks_layers)
            else:
                p_dense, p_dense_layers = denseinit()
                p_gene_para.append(p_dense)
                p_layers.append(p_dense_layers)

        elif len(item) == 1:
            if item in params.Choice_bn:
                p_bn, p_bn_layers = bninit()
                p_gene_para.append(p_bn)
                p_layers.append(p_bn_layers)
            elif item in params.Choice_af:
                p_af, p_af_layers = afinit()
                p_gene_para.append(p_af)
                p_layers.append(p_af_layers)
            elif item in params.Choice_fc:
                p_fc, p_fc_layers = fcinit()
                p_gene_para.append(p_fc)
                p_layers.append(p_fc_layers)
    p_gene = geneinit()
    p_operate = mutate_opinit()
    return p_gene, p_operate, p_layers, p_gene_para

def get_mutt_gene_pos(py_layers, mutt_gene):
    i = -1
    pos = []
    if mutt_gene == 'conv':
        for item in py_layers:
            i += 1
            if len(item) == 4:
                pos.append(i)

    elif mutt_gene == 'pool':
        for item in py_layers:
            i += 1
            if len(item) == 3 and item[2] in params.Choice_poolname:
                pos.append(i)

    elif mutt_gene == 'blocks':
        for item in py_layers:
            i += 1
            if len(item) == 3 and item[0] in params.Chioce_resblocks_type:
                pos.append(i)
    elif mutt_gene == 'dense':
        for item in py_layers:
            i += 1
            if len(item) == 3 and item[2] not in params.Choice_poolname and item[0] not in params.Chioce_resblocks_type:
                pos.append(i)

    elif mutt_gene == 'bn':
        for item in py_layers:
            i += 1
            if item in params.Choice_bn:
                pos.append(i)

    elif mutt_gene == 'af':
        for item in py_layers:
            i += 1
            if item in params.Choice_af:
                pos.append(i)

    elif mutt_gene == 'fc':
        for item in py_layers:
            i += 1
            if item in params.Choice_fc:
                pos.append(i)
    if len(pos) == 0:
        place_gene = -100
    else:
        place_gene = np.random.choice(pos)
    return place_gene

def get_muteta(p_layers, py_layers, p_gene, p_operate, p_gene_para):
    addp1 = 0
    global flag_fc
    flag_fc = 'true'
    for i in p_gene.values():
        addp1 = addp1 + i
    mutt_gene, place_gene_no = random_pick(p_gene.keys(), p_gene.values(), addp1)
    mutt_gene = str(mutt_gene)

    place_gene = get_mutt_gene_pos(py_layers, mutt_gene)

    some_list2 = ['add', 'diss', 'param']
    if place_gene == -100:
        mutt2 = 'add'
        place_operate = 0
    else:
        addp2 = 0
        for _ in p_operate:
            addp2 = addp2 + _
        mutt2, place_operate = random_pick(some_list2, p_operate, addp2)
    place_gene_para_s = -1
    place_layer_para = -1

    if mutt_gene == 'conv':
        pos = 0
        if mutt2 == 'add':
            pos = random.choice(range(len(py_layers)))
            while py_layers[pos - 1] in params.Choice_fc:
                pos = random.choice(range(len(py_layers)))
            if place_gene == -100:
                py_layers.insert(pos, params.B_conv)
                p_conv_gene, p_conv_layers = convinit()
                p_layers.insert(pos, p_conv_layers)
                p_gene_para.insert(pos, p_conv_gene)
            else:
                py_layers.insert(pos, py_layers[place_gene])
                p_layers.insert(pos, p_layers[place_gene])
                p_gene_para.insert(pos, p_gene_para[place_gene])
        elif mutt2 == 'diss':
            del py_layers[place_gene]
            del p_layers[place_gene]
            del p_gene_para[place_gene]
        elif mutt2 == 'param':
            addpconv = 0
            for _ in p_gene_para[place_gene]:
                addpconv = addpconv + _
            param_to_be_mutated_conv, iconv = random_pick(params.conv, p_gene_para[place_gene], addpconv)
            place_gene_para_s = iconv
            if param_to_be_mutated_conv == 'conv_kernal':
                addpconv = 0
                for _ in p_layers[place_gene][0]:
                    addpconv = addpconv + _
                param_to_be_mutated_conv_kernalsize, iconv_kernalsize = \
                    random_pick(params.Choice_kernal, p_layers[place_gene][0], addpconv)
                place_layer_para = iconv_kernalsize
                py_layers[place_gene][iconv] = param_to_be_mutated_conv_kernalsize
            elif param_to_be_mutated_conv == 'conv_number':
                addpconv = 0
                for _ in p_layers[place_gene][1]:
                    addpconv = addpconv + _
                param_to_be_mutated_conv_num, iconv_num = \
                    random_pick(params.Choice_channel, p_layers[place_gene][1], addpconv)
                place_layer_para = iconv_num
                py_layers[place_gene][iconv] = param_to_be_mutated_conv_num
            elif param_to_be_mutated_conv == 'conv_stride':
                addpconv = 0
                for _ in p_layers[place_gene][2]:
                    addpconv = addpconv + _
                param_to_be_mutated_conv_strides, iconv_strides = \
                    random_pick(params.Choice_stride, p_layers[place_gene][2], addpconv)
                place_layer_para = iconv_strides
                py_layers[place_gene][iconv] = param_to_be_mutated_conv_strides
            elif param_to_be_mutated_conv == 'conv_padding':
                addpconv = 0
                for _ in p_layers[place_gene][3]:
                    addpconv = addpconv + _
                param_to_be_mutated_conv_padding, iconv_padding = \
                    random_pick(params.Choice_padding, p_layers[place_gene][3], addpconv)
                place_layer_para = iconv_padding
                py_layers[place_gene][iconv] = param_to_be_mutated_conv_padding

    elif mutt_gene == 'pool':
        if mutt2 == 'add':
            pos = random.choice(range(len(py_layers)))
            while py_layers[pos - 1] in params.Choice_fc:
                pos = random.choice(range(len(py_layers)))
            if place_gene == -100:
                py_layers.insert(pos, params.B_pool)
                p_pool_gene, p_pool_layers = poolinit()
                p_layers.insert(pos, p_pool_layers)
                p_gene_para.insert(pos, p_pool_gene)

            else:
                py_layers.insert(pos, py_layers[place_gene])
                p_layers.insert(pos, p_layers[place_gene])
                p_gene_para.insert(pos, p_gene_para[place_gene])
        elif mutt2 == 'diss':
            del py_layers[place_gene]
            del p_layers[place_gene]
            del p_gene_para[place_gene]
        elif mutt2 == 'param':
            addpconv = 0
            for _ in p_gene_para[place_gene]:
                addpconv = addpconv + _
            param_to_be_mutated_pool, ipool = random_pick(params.pool, p_gene_para[place_gene], addpconv)
            place_gene_para_s = ipool
            if param_to_be_mutated_pool == 'pool_kernal':
                addpconv = 0
                for _ in p_layers[place_gene][0]:
                    addpconv = addpconv + _
                param_to_be_mutated_pool_kernalsize, ipool_kernalsize = \
                    random_pick(params.Choice_pool_kernal, p_layers[place_gene][0], addpconv)
                place_layer_para = ipool_kernalsize
                py_layers[place_gene][ipool] = param_to_be_mutated_pool_kernalsize
            elif param_to_be_mutated_pool == 'pool_stride':
                addpconv = 0
                for _ in p_layers[place_gene][1]:
                    addpconv = addpconv + _
                param_to_be_mutated_pool_stride, ipool_stride = \
                    random_pick(params.Choice_stride, p_layers[place_gene][1], addpconv)
                place_layer_para = ipool_stride
                py_layers[place_gene][ipool] = param_to_be_mutated_pool_stride
            elif param_to_be_mutated_pool == 'pool_type':
                addpconv = 0
                for _ in p_layers[place_gene][2]:
                    addpconv = addpconv + _
                param_to_be_mutated_pool_name, iblocks_channal = \
                    random_pick(params.Choice_poolname, p_layers[place_gene][2], addpconv)
                place_layer_para = iblocks_channal
                py_layers[place_gene][ipool] = param_to_be_mutated_pool_name

    elif mutt_gene == 'dense':
        if mutt2 == 'add':
            pos = random.choice(range(len(py_layers)))

            while py_layers[pos - 1] in params.Choice_fc:
                pos = random.choice(range(len(py_layers)))
            if place_gene == -100:
                py_layers.insert(pos, params.B_dense)
                p_dense_gene, p_dense_layers = denseinit()
                p_layers.insert(pos, p_dense_layers)
                p_gene_para.insert(pos, p_dense_gene)
            else:
                py_layers.insert(pos, py_layers[place_gene])
                p_layers.insert(pos, p_layers[place_gene])
                p_gene_para.insert(pos, p_gene_para[place_gene])
        elif mutt2 == 'diss':
            del py_layers[place_gene]
            del p_layers[place_gene]
            del p_gene_para[place_gene]
        elif mutt2 == 'param':
            addpconv = 0
            for _ in p_gene_para[place_gene]:
                addpconv = addpconv + _
            param_to_be_mutated_dense, idense = random_pick(params.dense_blocks, p_gene_para[place_gene], addpconv)
            place_gene_para_s = idense
            if param_to_be_mutated_dense == 'dense_k':
                addpconv = 0
                for _ in p_layers[place_gene][0]:
                    addpconv = addpconv + _
                param_to_be_mutated_dense_k, idense_k = \
                    random_pick(params.Chioce_dense_k, p_layers[place_gene][0], addpconv)
                place_layer_para = idense_k
                py_layers[place_gene][idense] = param_to_be_mutated_dense_k
            elif param_to_be_mutated_dense == 'dense_keranl':
                addpconv = 0
                for _ in p_layers[place_gene][1]:
                    addpconv = addpconv + _
                param_to_be_mutated_dense_kernel, idense_kernel = \
                    random_pick(params.Choice_kernal, p_layers[place_gene][1], addpconv)
                place_layer_para = idense_kernel
                py_layers[place_gene][idense] = param_to_be_mutated_dense_kernel
            elif param_to_be_mutated_dense == 'dense_number':
                addpconv = 0
                for _ in p_layers[place_gene][2]:
                    addpconv = addpconv + _
                param_to_be_mutated_dense_name, idense_num = \
                    random_pick(params.Chioce_dense_num, p_layers[place_gene][2], addpconv)
                place_layer_para = idense_num
                py_layers[place_gene][idense] = param_to_be_mutated_dense_name

    elif mutt_gene == 'blocks':
        if mutt2 == 'add':
            pos = random.choice(range(len(py_layers)))
            while py_layers[pos - 1] in params.Choice_fc:
                pos = random.choice(range(len(py_layers)))
            if place_gene == -1:
                py_layers.insert(pos, params.B_blocks)
                p_blocks_gene, p_blocks_layers = blocksinit()
                p_layers.insert(pos, p_blocks_layers)
                p_gene_para.insert(pos, p_blocks_gene)
            else:
                py_layers.insert(pos, py_layers[place_gene])
                p_layers.insert(pos, p_layers[place_gene])
                p_gene_para.insert(pos, p_gene_para[place_gene])
        elif mutt2 == 'diss':
            del py_layers[place_gene]
            del p_layers[place_gene]
            del p_gene_para[place_gene]
        elif mutt2 == 'param':
            addpconv = 0
            for _ in p_gene_para[place_gene]:
                addpconv = addpconv + _
            param_to_be_mutated_blocks, iblocks = random_pick(params.resnet_blocks, p_gene_para[place_gene], addpconv)
            place_gene_para_s = iblocks
            if param_to_be_mutated_blocks == 'res_type':
                addpconv = 0
                for _ in p_layers[place_gene][0]:
                    addpconv = addpconv + _
                param_to_be_mutated_blocks_type, iblocks_type = \
                    random_pick(params.Chioce_resblocks_type, p_layers[place_gene][0], addpconv)
                place_layer_para = iblocks_type
                py_layers[place_gene][iblocks] = param_to_be_mutated_blocks_type
            elif param_to_be_mutated_blocks == 'res_kernal':
                addpconv = 0
                for _ in p_layers[place_gene][1]:
                    addpconv = addpconv + _
                param_to_be_mutated_blocks_kernalsize, iblocks_kernalsize = \
                    random_pick(params.Choice_kernal, p_layers[place_gene][1], addpconv)
                place_layer_para = iblocks_kernalsize
                py_layers[place_gene][iblocks] = param_to_be_mutated_blocks_kernalsize
            elif param_to_be_mutated_blocks == 'res_repeat_times':
                addpconv = 0
                for _ in p_layers[place_gene][2]:
                    addpconv = addpconv + _
                param_to_be_mutated_blocks_num, iblocks_blocks_num = \
                    random_pick(params.Choice_block_num, p_layers[place_gene][2], addpconv)
                place_layer_para = iblocks_blocks_num
                py_layers[place_gene][iblocks] = param_to_be_mutated_blocks_num

    elif mutt_gene == 'bn':
        if mutt2 == 'add':
            pos = random.choice(range(len(py_layers)))
            while py_layers[pos - 1] in params.Choice_fc:
                pos = random.choice(range(len(py_layers)))
            if place_gene == -100:
                py_layers.insert(pos, params.B_bn)
                p_bn_gene, p_bn_layers = bninit()
                p_layers.insert(pos, p_bn_layers)
                p_gene_para.insert(pos, p_bn_gene)
            else:
                py_layers.insert(pos, py_layers[place_gene])
                p_layers.insert(pos, p_layers[place_gene])
                p_gene_para.insert(pos, p_gene_para[place_gene])
        elif mutt2 == 'diss':
            del py_layers[place_gene]
            del p_layers[place_gene]
            del p_gene_para[place_gene]

        elif mutt2 == 'param':
            addpconv = 0
            for _ in p_gene_para[place_gene]:
                addpconv = addpconv + _
            param_to_be_mutated_3_bn, i_3_bn = \
                random_pick(params.Choice_bn, p_gene_para[place_gene], addpconv)
            place_layer_para = i_3_bn
            place_gene_para_s = i_3_bn
            py_layers[place_gene] = param_to_be_mutated_3_bn
    elif mutt_gene == 'af':
        if mutt2 == 'add':
            pos = random.choice(range(len(py_layers)))
            while py_layers[pos - 1] in params.Choice_fc:
                pos = random.choice(range(len(py_layers)))
            if place_gene == -100:
                py_layers.insert(pos, params.B_af)
                p_af_gene, p_af_layers = afinit()
                p_layers.insert(pos, p_af_layers)
                p_gene_para.insert(pos, p_af_gene)
            else:
                py_layers.insert(pos, py_layers[place_gene])
                p_layers.insert(pos, p_layers[place_gene])
                p_gene_para.insert(pos, p_gene_para[place_gene])
        elif mutt2 == 'diss':
            del py_layers[place_gene]
            del p_layers[place_gene]
            del p_gene_para[place_gene]

        elif mutt2 == 'param':
            addpconv = 0
            for _ in p_gene_para[place_gene]:
                addpconv = addpconv + _
            param_to_be_mutated_3_af, i_3_af = random_pick(params.Choice_af, p_gene_para[place_gene], addpconv)
            place_layer_para = i_3_af
            place_gene_para_s = i_3_af
            py_layers[place_gene] = param_to_be_mutated_3_af
    elif mutt_gene == 'fc':
        if mutt2 == 'add':
            pos = random.choice(range(len(py_layers)))
            while len(py_layers[pos]) != 1 or py_layers[pos] not in params.Choice_fc:
                pos = random.choice(range(len(py_layers)))
            py_layers.insert(pos, py_layers[place_gene])
            p_layers.insert(pos, p_layers[place_gene])
            p_gene_para.insert(pos, p_gene_para[place_gene])
        elif mutt2 == 'diss':
            if place_gene + 1 == len(py_layers):
                flag_fc = 'false'
            else:
                del py_layers[place_gene]
                del p_layers[place_gene]
                del p_gene_para[place_gene]
        elif mutt2 == 'param':
            if place_gene + 1 == len(py_layers):
                place_gene_para_s = 0
                place_layer_para = 0
                py_layers[place_gene] = params.B_fc
            else:
                addpconv = 0
                for _ in p_gene_para[place_gene]:
                    addpconv = addpconv + _
                param_to_be_mutated_3_fc, i_3_fc = random_pick(params.Choice_fc, p_gene_para[place_gene], addpconv)
                place_gene_para_s = i_3_fc
                place_layer_para = i_3_fc
                py_layers[place_gene] = param_to_be_mutated_3_fc
    return py_layers, p_layers, p_gene, p_operate, p_gene_para, mutt_gene, place_gene, place_operate, \
           place_gene_para_s, place_layer_para, flag_fc