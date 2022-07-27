import numpy as np
import h5py
from scipy.linalg import sqrtm
import torch


#----------------------------------------------------------------------------------------------------------------------
# 取mat数据, 处理数据
def load_data_wrapper_train(file_name):
    datamat = h5py.File(file_name, 'r')
    # mat文件有很多不相关的keys，要找到目标的keys
    print(datamat.keys())  # 输出为dict_keys(['__header__', '__version__', '__globals__', 'data', 'label'])
    train_outputs = np.array(datamat['target_tout_train'])  # train_real_outputs是numpy.ndarray格式
    train_inputs = np.array(datamat['bk_train'])
    d = train_inputs.shape[1]
    train_inputs = train_inputs.reshape((90000, int(d**0.5), int(d**0.5)))
    train_inputs = np.expand_dims(train_inputs, 3)
    train_inputs = np.transpose(train_inputs, (0, 3, 1, 2))

    train_outputs = train_outputs.reshape((90000, int(d**0.5), int(d**0.5)))
    train_outputs = np.expand_dims(train_outputs, 3)
    train_outputs = np.transpose(train_outputs, (0, 3, 1, 2))

    return train_inputs, train_outputs


def load_data_wrapper_val(file_name):
    datamat2 = h5py.File(file_name, 'r')
    # mat文件有很多不相关的keys，要找到目标的keys
    print(datamat2.keys())  # 输出为dict_keys(['__header__', '__version__', '__globals__', 'data', 'label'])
    val_outputs = np.array(datamat2['target_tout_val'])  # val_real_outputs是numpy.ndarray格式
    val_inputs = np.array(datamat2['bk_val'])
    d = val_inputs.shape[1]
    train_inputs_val = val_inputs.reshape((10000, int(d**0.5), int(d**0.5)))
    train_inputs_val = np.expand_dims(train_inputs_val, 3)
    train_inputs_val = np.transpose(train_inputs_val, (0, 3, 1, 2))

    train_outputs_val = val_outputs.reshape((10000, int(d**0.5), int(d**0.5)))
    train_outputs_val = np.expand_dims(train_outputs_val, 3)
    train_outputs_val = np.transpose(train_outputs_val, (0, 3, 1, 2))

    return train_inputs_val, train_outputs_val


def load_data_wrapper_test(file_name):
    datamat3 = h5py.File(file_name, 'r')
    # mat文件有很多不相关的keys，要找到目标的keys
    print(datamat3.keys())  # 输出为dict_keys(['__header__', '__version__', '__globals__', 'data', 'label'])
    test_outputs = np.array(datamat3['target_tout_test'])  # val_real_outputs是numpy.ndarray格式
    test_inputs = np.array(datamat3['bk_test'])
    d = test_inputs.shape[1]
    train_inputs_test = test_inputs.reshape((1000, int(d**0.5), int(d**0.5)))
    train_inputs_test = np.expand_dims(train_inputs_test, 3)
    train_inputs_test = np.transpose(train_inputs_test, (0, 3, 1, 2))

    test_outputs = test_outputs.reshape((1000, int(d ** 0.5), int(d ** 0.5)))
    train_outputs_test = np.expand_dims(test_outputs, 3)
    train_outputs_test = np.transpose(train_outputs_test, (0, 3, 1, 2))

    return train_inputs_test, train_outputs_test





def generate_mask4(data, k):
    d = data.shape[0]
    mask = np.zeros((d, 16, 16))
    one_mask = np.zeros((16, 16))
    # one_mask = np.array(
    #     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # one_mask = np.array(
    #     [1, 1, 1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0])
    # one_mask = np.array(
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    one = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for j in range(k):
        one_mask[j] = one
    for i in range(d):
        mask[i] = one_mask
    return mask




#-----------------------------------------------------------------------------------------------------------------------
# 平均保真度
# 平均保真度
def fidelity_calcu(rho_pre, rho_true):

    d = rho_true.shape[0]
    fidelity_ave = 0
    for i in range(d):
        fidel = np.dot(np.dot(sqrtm(rho_true[i, :, :]), rho_pre[i, :, :]), sqrtm(rho_true[i, :, :]))  # 矩阵乘法
        fidelity = np.trace(sqrtm(fidel))
        fidelity_ave += fidelity/d
    return fidelity_ave
