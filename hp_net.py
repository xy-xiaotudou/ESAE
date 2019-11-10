# -*- encoding: utf8 -*-
def get_hp_net(Dataset):
    if Dataset == 'cifar10':
        py_layers = []
        py_layers.append([[3, 64, 2, 'SAME'], [10]])
        py_layers.append([[6, 3, 2], ['relu'], [10]])
        py_layers.append([['b', 2, 3], [2, 1, 'max'], [10]])
    elif Dataset == 'fashionmnist':
        py_layers = []
        py_layers.append([[3, 64, 2, 'SAME'], [2, 1, 'max'], [2, 1, 'max'], [10]])
        py_layers.append([[6, 3, 2], ['relu'], [10]])
        py_layers.append([['b', 2, 3], [2, 1, 'max'], [10]])

    print(py_layers)
    return py_layers

