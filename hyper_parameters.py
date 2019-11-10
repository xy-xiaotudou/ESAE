gene = ['conv', 'bn', 'pool', 'blocks', 'dense', 'af', 'fc']
conv = ['conv_kernal', 'conv_number', 'conv_padding', 'conv_stride']
bn = ['bn_have']
pool = ['pool_kernal', 'pool_stride', 'pool_type']
resnet_blocks = ['res_type', 'res_kernal', 'res_repeat_times']
dense_blocks = ['dense_k', 'dense_keranl', 'dense_number']
af = ['af_type']
fc = ['fc_number']
Choice_kernal = [1, 3, 5, 7] ## conv_kernal、res_kernal、dense_keranl
Choice_poolname = ['max', 'avg'] ## pool_type
Choice_pool_kernal = [1, 2]  ## pool_kernal
Choice_channel = [16, 32, 64, 128, 256]
Choice_stride = [1, 2] ## conv_stride、pool_stride
Choice_padding = ['SAME'] ## conv_padding
Choice_block_num = [1, 2, 3, 4, 5]  ## res_repeat_times
Chioce_resblocks_type = ['a', 'b']  ## res_type
Choice_bn = [['true'], ['false']]  ## bn_have
Chioce_dense_k = [6, 12, 24, 32, 36]  ## dense_k
Chioce_dense_num = [2, 3, 4, 5, 6]  ## dense_number
Choice_af = [['relu'], ['softmax'], ['sigmoid'], ['softplus'], ['tanh'], ['none']] ## af_type
Choice_fc = [[10], [50], [100], [128], [200]]  ## fc_number


## dafult paras
ACCURACY = 0.98 ##停止准确率
TIME = 100 #演化代数
ALL_MUTATE_NUM = 2  ##每个父本变异的子代个数
NUM_PARENT = 3    ##父本数量
MUTATE_TOP = 2    ##

lamda_gene = 0.95
lamda_others = 10

NUMBER_FOR_CHANGE_P = 2 ###每多少代改变基因概率分布

GENENUM = 7  ## number of gene
OPERATENUM = 3  ## number of mutate's operation


## dafult gene paras
flag_fc = 'true'
B_conv = [3, 64, 2, 'SAME']
B_pool = [2, 1, 'max']
B_bn = ['true']
B_af = ['relu']
B_fc = [10]
B_dense = [12, 3, 2]
B_blocks = [2, 3]
B_blocks = ['b', 2, 3]
