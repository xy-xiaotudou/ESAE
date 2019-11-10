# -*- encoding: utf8 -*-
import copy
import tensorflow as tf
import fit
import mutate
import csv
import pandas as pd
import numpy as np
import random
import hyper_parameters as paras
import hp_net

struct = []
struct_acc = []
excel_celname = ['gene', 'acc', 'p_gene', 'p_gene_para', 'p_operate', 'p_layers',
                     'k_conv', 'k_bn', 'k_pool', 'k_blocks',  'k_dense', 'k_af', 'k_fc', 'flag']
global Mutate_num
Mutate_num = 0

#--------------- Arguments Parser --------------
import argparse
parser = argparse.ArgumentParser(description='Parameters to be used for data training.')
parser.add_argument('--dataset', help='Input dataset file name', required=True)
parser.add_argument('--classes', help='Data classes', required=True)
args = parser.parse_args()

def mymain(g_name):
    global Dataset
    Dataset = args.dataset
    global Mutate_num
    out = open('result.csv', 'a', newline='')  ###存运行过程中所有的结果
    out.close()
    out = open('child.csv', 'a', newline='')   ###存变异与交叉的个体，以便选取下一代父本
    out.close()
    out = open('parent.csv', 'a', newline='')  ###存所有的父本
    out.close()
    out = open('mutate.csv', 'a', newline='')  ###存变异产生的个体
    out.close()
    out = open('cross.csv', 'a', newline='')   ###存交叉产生的个体
    out.close()
    out = open('two_one.csv', 'a', newline='')  ###存每个父本变异2个个体后，适应度高的一个
    out.close()
    out = open('parent.csv', 'r')
    # 设定读取模式
    reader = csv.reader(out)
    j = 0
    for item in reader:
        j += 1
    len_parent = j
    out.close()
    print('len_parent:', len_parent)
    if len_parent == 0:
        parent_hp = hp_net.get_hp_net(Dataset=Dataset)
        parent_acc, p_gene, p_operate, p_min_value, p_gene_para = [], [], [], [], []
        for i in range(0, paras.NUM_PARENT):
            parent_acc.append(fit.get_accuracy(parent_hp[i], g_name=get_graph(Mutate_num), Dataset=Dataset))
            tp_gene, tp_operate, tp_min_value, tp_gene_para = mutate.initmuteta(parent_hp[i])
            p_gene.append(tp_gene)
            p_operate.append(tp_operate)
            p_min_value.append(tp_min_value)
            p_gene_para.append(tp_gene_para)
            Mutate_num += 1

        k_conv, k_bn, k_pool, k_blocks, k_dense, k_af, k_fc = 0, 0, 0, 0, 0, 0, 0

        out = open('parent.csv', 'r')
        reader = csv.reader(out)
        j = 0
        for item in reader:
            j += 1
        len_csv = j
        out.close()
        if len_csv == 0:
            out = open('parent.csv', 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow(excel_celname)
            out.close()

        for i_par in range(0, paras.NUM_PARENT):
            out = open('parent.csv', 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow([parent_hp[i_par], parent_acc[i_par], p_gene[i_par],
                                p_gene_para[i_par], p_operate[i_par], p_min_value[i_par],
                                k_conv, k_bn, k_pool, k_blocks, k_dense, k_af, k_fc, 'parent'])
        out.close()

    st_parent_hp, parent_acc, p_gene, p_operate, p_min_value, p_gene_para, \
    k_conv, k_bn, k_pool, k_blocks, k_dense, k_af, k_fc, times = get_parent()

    best_p_acc = float(max(parent_acc))
    parent_hp = st_parent_hp
    while best_p_acc < paras.ACCURACY and times < paras.TIME:
        next_p = 0
        while next_p < len(parent_hp):
            get_mutate_child(parent_acc_next_p=parent_acc[next_p], parent_hp_next_p=parent_hp[next_p],
                             p_min_value_next_p=p_min_value[next_p], p_gene_next_p=p_gene[next_p],
                             p_operate_next_p=p_operate[next_p], p_gene_para_next_p=p_gene_para[next_p],
                             k_conv_next_p=k_conv[next_p],
                             k_bn_next_p=k_bn[next_p], k_pool_next_p=k_pool[next_p], k_blocks_next_p=k_blocks[next_p],
                             k_dense_next_p=k_dense[next_p], k_af_next_p=k_af[next_p], k_fc_next_p=k_fc[next_p])
            next_p += 1
        ##### cross
        cross_oprate()

        n_parent_hp, n_parent_acc, n_p_gene, n_p_operate, n_p_min_value, n_p_gene_para, \
        n_k_conv, n_k_bn, n_k_pool, n_k_blocks, n_k_dense, n_k_af, n_k_fc = get_next_parent()
        parent_hp = n_parent_hp
        parent_acc = n_parent_acc
        p_gene = n_p_gene
        p_operate = n_p_operate
        p_min_value = n_p_min_value
        p_gene_para = n_p_gene_para
        k_conv = n_k_conv
        k_bn = n_k_bn
        k_pool = n_k_pool
        k_blocks = n_k_blocks
        k_dense = n_k_dense
        k_af = n_k_af
        k_fc = n_k_fc
        best_p_acc = float(max(parent_acc))

        out = open('parent.csv', 'r')
        # 设定读取模式
        reader = csv.reader(out)
        j = 0
        for item in reader:
            j += 1
        len_parent = j
        out.close()
        times = int((len_parent - 1) / paras.NUM_PARENT)

def get_cross_pos(gene1, gene2):
    pos_list = min(len(eval(gene1[0])), len(eval(gene2[0])))
    pos = random.choice(range(pos_list))
    return pos

def make_cross_is_true(gene1, gene2):
    global Mutate_num
    pos = get_cross_pos(gene1, gene2)
    newgene1_hp,newgene1_p_gene, newgene1_p_gene_para, newgene1_p_operate, newgene1_p_min_value, \
    newgene2_hp, newgene2_p_gene, newgene2_p_gene_para, newgene2_p_operate, newgene2_p_min_value \
                              = get_new_cross_gene(gene1, gene2, pos)
    pool_flag1 = fit.if_pool(newgene1_hp)
    pool_flag2 = fit.if_pool(newgene2_hp)
    while pool_flag1 == 'false' or pool_flag2 == 'false':
        pos = get_cross_pos(gene1, gene2)
        newgene1_hp, newgene1_p_gene, newgene1_p_gene_para, newgene1_p_operate, newgene1_p_min_value, \
        newgene2_hp, newgene2_p_gene, newgene2_p_gene_para, newgene2_p_operate, newgene2_p_min_value \
            = get_new_cross_gene(gene1, gene2, pos)
        pool_flag1 = fit.if_pool(newgene1_hp)
        pool_flag2 = fit.if_pool(newgene2_hp)

    newgene1_acc = fit.get_accuracy(newgene1_hp, g_name=get_graph(Mutate_num), Dataset=Dataset)
    Mutate_num += 1
    newgene2_acc = fit.get_accuracy(newgene2_hp, g_name=get_graph(Mutate_num), Dataset=Dataset)
    Mutate_num += 1

    out = open('cross.csv', 'r')
    reader = csv.reader(out)
    j = 0
    for item in reader:
        j += 1
    len_csv = j
    out.close()
    if len_csv == 0:
        out = open('cross.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(excel_celname)
        out.close()
    out = open('cross.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow([newgene1_hp, newgene1_acc, newgene1_p_gene, newgene1_p_gene_para,
                        newgene1_p_operate, newgene1_p_min_value,
                        0, 0, 0, 0, 0, 0, 0, 'cross'])
    csv_write.writerow([newgene2_hp, newgene2_acc, newgene2_p_gene, newgene2_p_gene_para,
                        newgene2_p_operate, newgene2_p_min_value,
                        0, 0, 0, 0, 0, 0, 0, 'cross'])
    out.close()

    out = open('result.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow([newgene1_hp, newgene1_acc, newgene1_p_gene, newgene1_p_gene_para,
                        newgene1_p_operate, newgene1_p_min_value,
                        0, 0, 0, 0, 0, 0, 0, 'cross'])
    csv_write.writerow([newgene2_hp, newgene2_acc, newgene2_p_gene, newgene2_p_gene_para,
                        newgene2_p_operate, newgene2_p_min_value,
                        0, 0, 0, 0, 0, 0, 0, 'cross'])
    out.close()
    out = open('child.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow([newgene1_hp, newgene1_acc, newgene1_p_gene, newgene1_p_gene_para,
                        newgene1_p_operate, newgene1_p_min_value,
                        0, 0, 0, 0, 0, 0, 0, 'cross'])
    csv_write.writerow([newgene2_hp, newgene2_acc, newgene2_p_gene, newgene2_p_gene_para,
                        newgene2_p_operate, newgene2_p_min_value,
                        0, 0, 0, 0, 0, 0, 0, 'cross'])
    out.close()

def cross_oprate():
    out = open('two_one.csv', 'r')
    reader = csv.reader(out)
    j = 0
    for item in reader:
        j += 1
    len_two_one = j
    out.close()
    data_child = pd.read_csv('two_one.csv')
    data_child_choice = data_child.loc[len_two_one-4:len_two_one-1]
    data_arr = np.array(data_child_choice)
    data_list = data_arr.tolist()
    gene_all = []
    for rows in data_list:
        gene_all.append(rows)

    gene1 = gene_all[0]
    gene2 = gene_all[1]
    gene3 = gene_all[2]
    make_cross_is_true(gene1, gene2)
    make_cross_is_true(gene1, gene3)
    make_cross_is_true(gene2, gene3)

def get_new_cross_gene(gene1, gene2, pos):
    newgene1_hp, newgene2_hp = [], []
    newgene1_p_gene, newgene2_p_gene = eval(gene1[2]), eval(gene1[2])
    newgene1_p_operate, newgene2_p_operate = [], []
    newgene1_p_min_value, newgene2_p_min_value = [], []
    newgene1_p_gene_para, newgene2_p_gene_para = [], []
    i, j, k, m = 0, 0, 0, 0
    for item1 in eval(gene1[0]):
        if i < pos:
            newgene1_hp.append(item1)
        i += 1
    for item2 in eval(gene2[0]):
        if j < pos:
            newgene2_hp.append(item2)
        j += 1
    for item in eval(gene1[0]):
        if k >= pos:
            newgene2_hp.append(item)
        k += 1
    for item2 in eval(gene2[0]):
        if m >= pos:
            newgene1_hp.append(item2)
        m += 1

    i, j, k, m = 0, 0, 0, 0
    for item1 in eval(gene1[4]):
        if i < pos:
            newgene1_p_operate.append(item1)
        i += 1
    for item2 in eval(gene2[4]):
        if j < pos:
            newgene2_p_operate.append(item2)
        j += 1
    for item1 in eval(gene1[4]):
        if k >= pos:
            newgene2_p_operate.append(item1)
        k += 1
    for item2 in eval(gene2[4]):
        if m >= pos:
            newgene1_p_operate.append(item2)
        m += 1

    i, j, k, m = 0, 0, 0, 0
    for item1 in eval(gene1[3]):
        if i < pos:
            newgene1_p_gene_para.append(item1)
        i += 1
    for item2 in eval(gene2[3]):
        if j < pos:
            newgene2_p_gene_para.append(item2)
        j += 1
    for item1 in eval(gene1[3]):
        if k >= pos:
            newgene2_p_gene_para.append(item1)
        k += 1
    for item2 in eval(gene2[3]):
        if m >= pos:
            newgene1_p_gene_para.append(item2)
        m += 1

    i, j, k, m = 0, 0, 0, 0
    global Mutate_num
    for item1 in eval(gene1[5]):
        if i < pos:
            newgene1_p_min_value.append(item1)
        i += 1
    for item2 in eval(gene2[5]):
        if j < pos:
            newgene2_p_min_value.append(item2)
        j += 1
    for item1 in eval(gene1[5]):
        if k >= pos:
            newgene2_p_min_value.append(item1)
        k += 1
    for item2 in eval(gene2[5]):
        if m >= pos:
            newgene1_p_min_value.append(item2)
        m += 1

    return newgene1_hp, newgene1_p_gene, newgene1_p_gene_para, newgene1_p_operate, newgene1_p_min_value, \
           newgene2_hp, newgene2_p_gene, newgene2_p_gene_para, newgene2_p_operate,  newgene2_p_min_value

def get_cross_child():
    out = open('mutate.csv', 'r')
    reader = csv.reader(out)
    j = 0
    for item in reader:
        j += 1
    len_child = j
    out.close()
    data_mutate_child = pd.read_csv('mutate.csv')
    data_mutate_child_choice = data_mutate_child.loc[len_child - 1 -
                                                     (paras.NUM_PARENT * paras.ALL_MUTATE_NUM):len_child - 1]
    data_mutate_child_choice.sort_values('acc', inplace=True, ascending=False)
    data_arr = np.array(data_mutate_child_choice)
    data_list = data_arr.tolist()
    i = 0
    mutate_child_all = []
    for rows in data_list:
        if i >= 0 and i < paras.MUTATE_TOP:
            mutate_child_all.append(rows)
        i += 1
    data_i = 0
    mutate_child_hp, mutate_child_acc, \
    mutate_child_p_gene, mutate_child_p_operate, mutate_child_p_min_value,\
    mutate_child_p_gene_para = [], [], [], [], [], []
    mutate_child_k_conv, mutate_child_k_bn, mutate_child_k_pool, \
    mutate_child_k_blocks, mutate_child_k_dense, mutate_child_k_af, mutate_child_k_fc = [], [], [], [], [], [], []
    while data_i < len(mutate_child_all):
        mutate_child_hp.append(eval(mutate_child_all[data_i][0]))
        mutate_child_acc.append(mutate_child_all[data_i][1])
        mutate_child_p_gene.append(eval(mutate_child_all[data_i][2]))
        mutate_child_p_operate.append(eval(mutate_child_all[data_i][4]))
        mutate_child_p_min_value.append(eval(mutate_child_all[data_i][5]))
        mutate_child_p_gene_para.append(eval(mutate_child_all[data_i][3]))
        mutate_child_k_conv.append(mutate_child_all[data_i][6])
        mutate_child_k_bn.append(mutate_child_all[data_i][7])
        mutate_child_k_pool.append(mutate_child_all[data_i][8])
        mutate_child_k_blocks.append(mutate_child_all[data_i][9])
        mutate_child_k_dense.append(mutate_child_all[data_i][10])
        mutate_child_k_af.append(mutate_child_all[data_i][11])
        mutate_child_k_fc.append(mutate_child_all[data_i][12])
        data_i += 1
    pos1 = get_cross_pos(mutate_child_hp[0])
    pos2 = get_cross_pos(mutate_child_hp[2])
    cross_oprate(gene=mutate_child_all, pos1=pos1, pos2=pos2)

def save_two_best_one():
    out = open('mutate.csv', 'r')
    reader = csv.reader(out)
    j = 0
    len_child = 0
    for item in reader:
        j += 1
    len_child = j
    out.close()
    data_child = pd.read_csv('mutate.csv')
    data_child_choice = data_child.loc[len_child - 1 - paras.MUTATE_TOP:len_child - 1]
    data_child_choice.sort_values('acc', inplace=True, ascending=False)
    data_arr = np.array(data_child_choice)
    data_list = data_arr.tolist()

    i = 0

    cross_gene_all = []
    for rows in data_list:
        if i == 0:
            cross_gene_all.append(rows)
        i += 1

    gene_hp = eval(cross_gene_all[0][0])
    gene_acc = cross_gene_all[0][1]
    gene_p_gene = eval(cross_gene_all[0][2])
    gene_p_operate = eval(cross_gene_all[0][4])
    gene_p_min_value = eval(cross_gene_all[0][5])
    gene_p_gene_para = eval(cross_gene_all[0][3])
    gene_k_conv = cross_gene_all[0][6]
    gene_k_bn = cross_gene_all[0][7]
    gene_k_pool = cross_gene_all[0][8]
    gene_k_blocks = cross_gene_all[0][9]
    gene_k_dense = cross_gene_all[0][10]
    gene_k_af = cross_gene_all[0][11]
    gene_k_fc = cross_gene_all[0][12]

    out = open('two_one.csv', 'r')
    reader = csv.reader(out)
    j = 0
    for item in reader:
        j += 1
    len_csv = j
    out.close()
    if len_csv == 0:
        out = open('two_one.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow(excel_celname)
        out.close()
    out = open('two_one.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow([gene_hp, gene_acc, gene_p_gene, gene_p_gene_para, gene_p_operate, gene_p_min_value,
                        gene_k_conv, gene_k_bn, gene_k_pool,
                        gene_k_blocks, gene_k_dense, gene_k_af, gene_k_fc])
    out.close()
    out = open('result.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow([gene_hp, gene_acc, gene_p_gene, gene_p_gene_para, gene_p_operate, gene_p_min_value,
                        gene_k_conv, gene_k_bn, gene_k_pool,
                        gene_k_blocks, gene_k_dense, gene_k_af, gene_k_fc, 'two-one'])
    out.close()

def get_mutate_child(parent_acc_next_p, parent_hp_next_p, p_min_value_next_p,
                    p_gene_next_p, p_operate_next_p, p_gene_para_next_p,
                     k_conv_next_p, k_bn_next_p, k_pool_next_p, k_blocks_next_p,
                     k_dense_next_p, k_af_next_p, k_fc_next_p):
    global Mutate_num
    best_acc = float(parent_acc_next_p)
    best_hp = copy.deepcopy(parent_hp_next_p)
    best_p_min_value = copy.deepcopy(p_min_value_next_p)
    best_p_gene = copy.deepcopy(p_gene_next_p)
    best_p_operate = copy.deepcopy(p_operate_next_p)
    best_p_gene_para = copy.deepcopy(p_gene_para_next_p)
    out = open('result.csv', 'r')
    reader = csv.reader(out)
    a = 0

    for item in reader:
        a += 1
    len_csv = a
    out.close()
    out = open('result.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    if len_csv == 0:
        csv_write.writerow(excel_celname)
        out.close()
    out = open('result.csv', 'a', newline='')

    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow([parent_hp_next_p, parent_acc_next_p, p_gene_next_p,
                        p_gene_para_next_p, p_operate_next_p, p_min_value_next_p,
                        k_conv_next_p, k_bn_next_p, k_pool_next_p,
                        k_blocks_next_p, k_dense_next_p, k_af_next_p, k_fc_next_p, 'parent'])
    out.close()
    mu_num = 0

    child_hp, child_p, cp_gene, cp_operate, cp_gene_para, child_acc = [], [], [], [], [], []
    while mu_num < paras.ALL_MUTATE_NUM:
        tem_p = copy.deepcopy(best_p_min_value)
        tem_hp = copy.deepcopy(best_hp)
        tem_p_gene = copy.deepcopy(best_p_gene)
        tem_p_operate = copy.deepcopy(best_p_operate)
        tem_p_gene_para = copy.deepcopy(best_p_gene_para)
        c_tem_p = copy.deepcopy(tem_p)
        c_tem_hp = copy.deepcopy(tem_hp)
        c_tem_p_gene = copy.deepcopy(tem_p_gene)
        c_tem_p_operate = copy.deepcopy(tem_p_operate)
        c_tem_p_gene_para = copy.deepcopy(tem_p_gene_para)
        t_child_hp, t_child_p, t_cp_gene, t_cp_operate, t_cp_gene_para, \
        mutt_gene, mutate_gene, mutate_op, mutate_gene_para, mutate_value, cis_fc_flag = \
            mutate.get_muteta(c_tem_p, c_tem_hp, c_tem_p_gene,
                              c_tem_p_operate, c_tem_p_gene_para)
        is_fc_flag = cis_fc_flag
        pool_flag = fit.if_pool(t_child_hp)
        block_flaf = fit.is_block(t_child_hp)

        while pool_flag == 'false' or is_fc_flag == 'false' or block_flaf == 'false':
            c_tem_p = copy.deepcopy(tem_p)
            c_tem_hp = copy.deepcopy(tem_hp)
            c_tem_p_gene = copy.deepcopy(tem_p_gene)
            c_tem_p_operate = copy.deepcopy(tem_p_operate)
            c_tem_p_gene_para = copy.deepcopy(tem_p_gene_para)
            t_child_hp, t_child_p, t_cp_gene, t_cp_operate, t_cp_gene_para, \
            mutt_gene, mutate_gene, mutate_op, mutate_gene_para, mutate_value, cis_fc_flag = \
                mutate.get_muteta(c_tem_p, c_tem_hp, c_tem_p_gene,
                                  c_tem_p_operate, c_tem_p_gene_para)
            c_tem_hp = copy.deepcopy(tem_hp)
            is_fc_flag = cis_fc_flag
            pool_flag = fit.if_pool(t_child_hp)
            block_flaf = fit.is_block(t_child_hp)

        child_hp.append(t_child_hp)
        child_p.append(t_child_p)
        cp_gene.append(t_cp_gene)
        cp_operate.append(t_cp_operate)
        cp_gene_para.append(t_cp_gene_para)

        t_child_acc = fit.get_accuracy(child_hp[mu_num], g_name=get_graph(Mutate_num), Dataset=Dataset)
        child_acc.append(t_child_acc)

        out = open('child.csv', 'r')
        reader = csv.reader(out)
        j = 0
        for item in reader:
            j += 1
        len_csv = j
        out.close()
        if len_csv == 0:
            out = open('child.csv', 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow(excel_celname)
            out.close()
        if child_acc[mu_num] > best_acc:

            child_p[mu_num], cp_operate[mu_num], cp_gene_para[mu_num] = \
                Modify_para(mutt_gene, mutate_gene, mutate_op, mutate_gene_para, mutate_value,
                            child_p[mu_num], cp_operate[mu_num], cp_gene_para[mu_num], '+')
            flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense, flag_af, flag_fc = \
                Modify_flag(mutt_gene, '+')

        else:
            child_p[mu_num], cp_operate[mu_num], cp_gene_para[mu_num] = \
                Modify_para(mutt_gene, mutate_gene, mutate_op, mutate_gene_para, mutate_value,
                            child_p[mu_num], cp_operate[mu_num], cp_gene_para[mu_num], '-')
            flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense, flag_af, flag_fc = \
                Modify_flag(mutt_gene, '-')

        out = open('result.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([child_hp[mu_num], child_acc[mu_num], cp_gene[mu_num], cp_gene_para[mu_num],
                            cp_operate[mu_num], child_p[mu_num],
                            flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense, flag_af, flag_fc])
        out.close()

        out = open('child.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([child_hp[mu_num], child_acc[mu_num], cp_gene[mu_num], cp_gene_para[mu_num],
                            cp_operate[mu_num], child_p[mu_num],
                            flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense, flag_af, flag_fc])
        out.close()

        out = open('mutate.csv', 'r')

        reader = csv.reader(out)
        j = 0
        for item in reader:
            j += 1
        len_csv = j
        out.close()
        if len_csv == 0:
            out = open('mutate.csv', 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow(excel_celname)
            out.close()
        out = open('mutate.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([child_hp[mu_num], child_acc[mu_num], cp_gene[mu_num], cp_gene_para[mu_num],
                            cp_operate[mu_num], child_p[mu_num],
                            flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense, flag_af, flag_fc])
        out.close()

        Mutate_num = Mutate_num + 1
        mu_num += 1
    save_two_best_one()

def get_parent():
    parent_all, parent_hp, parent_acc, \
    p_gene, p_operate, p_min_value, p_gene_para = [], [], [], [], [], [], []
    k_conv, k_bn, k_pool, k_blocks, k_dense, k_af, k_fc = [], [], [], [], [], [], []
    out = open('parent.csv', 'r')
    reader = csv.reader(out)
    j = 0
    for item in reader:
        j += 1
    len_parent = j
    out.close()
    out = open('parent.csv', 'r')

    reader = csv.reader(out)
    for i, rows in enumerate(reader):
        if i >= len_parent - paras.NUM_PARENT and i <= len_parent - 1:
            parent_all.append(rows)
    out.close()

    data_i = 0
    while data_i < len(parent_all):
        parent_hp.append(eval(parent_all[data_i][0]))
        parent_acc.append(parent_all[data_i][1])
        p_gene.append(eval(parent_all[data_i][2]))
        p_operate.append(eval(parent_all[data_i][4]))
        p_min_value.append(eval(parent_all[data_i][5]))
        p_gene_para.append(eval(parent_all[data_i][3]))
        k_conv.append(parent_all[data_i][6])
        k_bn.append(parent_all[data_i][7])
        k_pool.append(parent_all[data_i][8])
        k_blocks.append(parent_all[data_i][9])
        k_dense.append(parent_all[data_i][10])
        k_af.append(parent_all[data_i][11])
        k_fc.append(parent_all[data_i][12])
        data_i += 1

    times = int((len_parent-1)/paras.NUM_PARENT)
    return parent_hp, parent_acc, p_gene, p_operate, p_min_value, p_gene_para, \
           k_conv, k_bn, k_pool, k_blocks, k_dense, k_af, k_fc, times

def get_next_parent():
    out = open('child.csv', 'r')

    reader = csv.reader(out)
    j = 0
    len_child = 0
    for item in reader:
        j += 1
        len_child = j
    out.close()

    data_child = pd.read_csv('child.csv')
    data_child_choice = data_child.loc[len_child - 1 - 12:len_child - 1]
    data_child_choice.sort_values('acc', inplace=True, ascending=False)
    data_arr = np.array(data_child_choice)
    data_list = data_arr.tolist()
    i = 0
    parent_all = []
    for rows in data_list:
        if i >= 0 and i < 2:
            parent_all.append(rows)
        i += 1

    out = open('parent.csv', 'r')
    reader = csv.reader(out)
    j = 0
    len_child = 0
    for item in reader:
        j += 1
        len_child = j
    out.close()

    data_child = pd.read_csv('parent.csv')
    data_child_choice = data_child.loc[len_child - 1 - 3:len_child - 1]

    data_child_choice.sort_values('acc', inplace=True, ascending=False)

    data_arr = np.array(data_child_choice)

    data_list = data_arr.tolist()
    i = 0
    for rows in data_list:
        if i == 0:
            parent_all.append(rows)
        i += 1
    data_i = 0
    n_parent_hp, n_parent_acc, \
    n_p_gene, n_p_operate, n_p_min_value, n_p_gene_para = [], [], [], [], [], []
    n_k_conv, n_k_bn, n_k_pool, n_k_blocks, n_k_dense, n_k_af, n_k_fc = [], [], [], [], [], [], []
    while data_i < len(parent_all):
        n_parent_hp.append(eval(parent_all[data_i][0]))
        n_parent_acc.append(parent_all[data_i][1])
        n_p_gene.append(eval(parent_all[data_i][2]))
        n_p_operate.append(eval(parent_all[data_i][4]))
        n_p_min_value.append(eval(parent_all[data_i][5]))
        n_p_gene_para.append(eval(parent_all[data_i][3]))
        n_k_conv.append(parent_all[data_i][6])
        n_k_bn.append(parent_all[data_i][7])
        n_k_pool.append(parent_all[data_i][8])
        n_k_blocks.append(parent_all[data_i][9])
        n_k_dense.append(parent_all[data_i][10])
        n_k_af.append(parent_all[data_i][11])
        n_k_fc.append(parent_all[data_i][12])
        data_i += 1

    out = open('parent.csv', 'r')

    reader = csv.reader(out)
    j = 0
    len_csv = 0
    for item in reader:
        j += 1
    len_csv = j
    out.close()
    co_p_gene = n_p_gene[0]

    if (len_csv-1) % (paras.NUMBER_FOR_CHANGE_P * paras.NUM_PARENT) == 0:
        co_p_gene = get_p_gene(n_p_gene[0])

    for i_par in range(0, len(n_parent_hp)):
        out = open('parent.csv', 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        csv_write.writerow([n_parent_hp[i_par], n_parent_acc[i_par], co_p_gene,
                            n_p_gene_para[i_par], n_p_operate[i_par], n_p_min_value[i_par],
                            n_k_conv[i_par], n_k_bn[i_par], n_k_pool[i_par],
                            n_k_blocks[i_par], n_k_dense[i_par], n_k_af[i_par], n_k_fc[i_par], 'parent'])
    out.close()
    parent_hp, parent_acc, p_gene, p_operate, p_min_value, p_gene_para, \
    k_conv, k_bn, k_pool, k_blocks, k_dense, k_af, k_fc, time = get_parent()
    return parent_hp, parent_acc, p_gene, p_operate, p_min_value, p_gene_para, \
           k_conv, k_bn, k_pool, k_blocks, k_dense, k_af, k_fc

def get_graph(Mutate_num):
    g_name = 'garph%d' % Mutate_num
    g_name = tf.Graph()
    return g_name

def get_p_gene(p_gene):
    out = open('mutate.csv', 'r')
    reader = csv.reader(out)
    len_csv = 0
    for i in reader:
        len_csv += 1
    out.close()
    p_gene_old = {}
    p_gene_old['conv'] = 1/paras.GENENUM
    p_gene_old['bn'] = 1/paras.GENENUM
    p_gene_old['pool'] = 1/paras.GENENUM
    p_gene_old['blocks'] = 1/paras.GENENUM
    p_gene_old['dense'] = 1/paras.GENENUM
    p_gene_old['af'] = 1/paras.GENENUM
    p_gene_old['fc'] = 1/paras.GENENUM
    out = open('mutate.csv', 'r')

    reader = csv.reader(out)
    flag_conv = [row[6] for row in reader]
    out.close()
    sum_flag_conv = 0
    i_conv = -1
    for item in flag_conv:
        i_conv += 1
        if item == '1' and i_conv>=len_csv-(paras.NUM_PARENT*paras.ALL_MUTATE_NUM)*paras.NUMBER_FOR_CHANGE_P \
                and i_conv<=len_csv:
            sum_flag_conv += 1

    out = open('mutate.csv', 'r')
    reader = csv.reader(out)
    flag_bn = [row[7] for row in reader]

    out.close()
    sum_flag_bn = 0
    i_bn = -1
    for item in flag_bn:
        i_bn += 1
        if item == '1' and i_conv >= len_csv - (paras.NUM_PARENT * paras.ALL_MUTATE_NUM) * paras.NUMBER_FOR_CHANGE_P\
                and i_conv <= len_csv:
            sum_flag_bn += 1

    out = open('mutate.csv', 'r')

    reader = csv.reader(out)
    flag_pool = [row[8] for row in reader]

    out.close()
    sum_flag_pool = 0
    i_pool = -1
    for item in flag_pool:
        i_pool += 1
        if item == '1' and i_conv >= len_csv - (paras.NUM_PARENT * Mutate_num) * paras.NUMBER_FOR_CHANGE_P\
                and i_conv <= len_csv:
            sum_flag_pool += 1

    out = open('mutate.csv', 'r')

    reader = csv.reader(out)
    flag_blocks = [row[9] for row in reader]

    out.close()
    sum_flag_blocks = 0
    i_blocks = -1
    for item in flag_blocks:
        i_blocks += 1
        if item == '1' and i_conv >= len_csv - (paras.NUM_PARENT * Mutate_num) * paras.NUMBER_FOR_CHANGE_P\
                and i_conv <= len_csv:
            sum_flag_blocks += 1

    out = open('mutate.csv', 'r')

    reader = csv.reader(out)
    flag_dense = [row[10] for row in reader]

    out.close()
    sum_flag_dense = 0
    i_dense = -1
    for item in flag_dense:
        i_dense += 1
        if item == '1' and i_conv >= len_csv - (paras.NUM_PARENT * Mutate_num) * paras.NUMBER_FOR_CHANGE_P\
                and i_conv <= len_csv:
            sum_flag_dense += 1

    out = open('mutate.csv', 'r')

    reader = csv.reader(out)
    flag_af = [row[11] for row in reader]
    out.close()
    sum_flag_af = 0
    i_af = -1
    for item in flag_af:
        i_af += 1
        if item == '1' and i_conv >= len_csv - (paras.NUM_PARENT * Mutate_num) * paras.NUMBER_FOR_CHANGE_P\
                and i_conv <= len_csv:
            sum_flag_af += 1

    out = open('mutate.csv', 'r')

    reader = csv.reader(out)
    flag_fc = [row[12] for row in reader]
    out.close()
    sum_flag_fc = 0
    i_fc = -1
    for item in flag_fc:
        i_fc += 1
        if item == '1' and i_conv >= len_csv - (paras.NUM_PARENT * Mutate_num) * paras.NUMBER_FOR_CHANGE_P\
                and i_conv <= len_csv:
            sum_flag_fc += 1

    sum = sum_flag_conv + sum_flag_bn + sum_flag_pool + sum_flag_blocks + sum_flag_dense + sum_flag_af + sum_flag_fc
    if sum != 0:
        m_p_conv = sum_flag_conv / sum
        m_p_bn = sum_flag_bn / sum
        m_p_pool = sum_flag_pool / sum
        m_p_blocks = sum_flag_blocks / sum
        m_p_dense = sum_flag_dense / sum
        m_p_af = sum_flag_af / sum
        m_p_fc = sum_flag_fc / sum
        mp_gene = {}
        mp_gene['conv'] = paras.lamda_gene * p_gene_old['conv'] + (1 - paras.lamda_gene) * m_p_conv
        mp_gene['bn'] = paras.lamda_gene * p_gene_old['bn'] + (1 - paras.lamda_gene) * m_p_bn
        mp_gene['pool'] = paras.lamda_gene * p_gene_old['pool'] + (1 - paras.lamda_gene) * m_p_pool
        mp_gene['blocks'] = paras.lamda_gene * p_gene_old['blocks'] + (1 - paras.lamda_gene) * m_p_blocks
        mp_gene['dense'] = paras.lamda_gene * p_gene_old['dense'] + (1 - paras.lamda_gene) * m_p_dense
        mp_gene['af'] = paras.lamda_gene * p_gene_old['af'] + (1 - paras.lamda_gene) * m_p_af
        mp_gene['fc'] = paras.lamda_gene * p_gene_old['fc'] + (1 - paras.lamda_gene) * m_p_fc
    else:
        mp_gene = {}
        mp_gene['conv'] = 1 / paras.GENENUM
        mp_gene['bn'] = 1 / paras.GENENUM
        mp_gene['pool'] = 1 / paras.GENENUM
        mp_gene['blocks'] = 1 / paras.GENENUM
        mp_gene['dense'] = 1 / paras.GENENUM
        mp_gene['af'] = 1 / paras.GENENUM
        mp_gene['fc'] = 1 / paras.GENENUM
    return mp_gene

def Modify_para(mutt_gene, mutate_gene, mutate_op, mutate_gene_para, mutate_value,
                p_layer, p_operate, p_gene_para, op):
    if op == '+':
        if mutate_op == 0:  # add
            p_operate[mutate_op] = p_operate[mutate_op]+p_operate[mutate_op] / paras.lamda_others

        elif mutate_op == 1:  # diss
            p_operate[mutate_op] = p_operate[mutate_op] + p_operate[mutate_op] / paras.lamda_others

        else:
            p_operate[mutate_op] = p_operate[mutate_op] + p_operate[mutate_op] / paras.lamda_others
            p_layer[mutate_gene][mutate_gene_para][mutate_value] = \
                p_layer[mutate_gene][mutate_gene_para][mutate_value] + \
                p_layer[mutate_gene][mutate_gene_para][mutate_value] / paras.lamda_others

    else:
        if mutate_op == 0:  # add
            p_operate[mutate_op] = p_operate[mutate_op]- p_operate[mutate_op] / paras.lamda_others

        elif mutate_op == 1:  # diss
            p_operate[mutate_op] = p_operate[mutate_op] - p_operate[mutate_op] / paras.lamda_others

        else:
            p_operate[mutate_op] = p_operate[mutate_op] - p_operate[mutate_op] / paras.lamda_others

            p_layer[mutate_gene][mutate_gene_para][mutate_value] = \
                p_layer[mutate_gene][mutate_gene_para][mutate_value] - \
                p_layer[mutate_gene][mutate_gene_para][mutate_value] / paras.lamda_others
    return p_layer, p_operate, p_gene_para

def Modify_flag(mutt_gene, op):
    flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense, flag_af, flag_fc = 0, 0, 0, 0, 0, 0, 0
    if op == '+':
        if mutt_gene == 'conv':
            flag_conv = 1
        elif mutt_gene == 'bn':
            flag_bn = 1
        elif mutt_gene == 'blocks':
            flag_blocks = 1
        elif mutt_gene == 'dense':
            flag_dense = 1
        elif mutt_gene == 'pool':
            flag_pool = 1
        elif mutt_gene =='af':
            flag_af = 1
        elif mutt_gene =='fc':
            flag_fc = 1
    else:
        flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense,  flag_af, flag_fc = 0, 0, 0, 0, 0, 0, 0
    return flag_conv, flag_bn, flag_pool, flag_blocks, flag_dense, flag_af, flag_fc

mymain(g_name=get_graph(Mutate_num))