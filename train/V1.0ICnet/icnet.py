from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import numpy as np
import sys


def conv(input,
         k_h,
         k_w,
         c_o,
         s_h,
         s_w,
         relu=False,
         padding="VALID",
         biased=False,
         name=None):
    act = None
    tmp = input
    if relu:
        act = "relu"
    if padding == "SAME":
        padding_h = max(k_h - s_h, 0)
        padding_w = max(k_w - s_w, 0)
        padding_top = padding_h // 2
        padding_left = padding_w // 2
        padding_bottom = padding_h - padding_top
        padding_right = padding_w - padding_left
        padding = [
            0, 0, 0, 0, padding_top, padding_bottom, padding_left, padding_right
        ]
        tmp = fluid.layers.pad(tmp, padding)
    tmp = fluid.layers.conv2d(
        tmp,
        num_filters=c_o,
        filter_size=[k_h, k_w],
        stride=[s_h, s_w],
        groups=1,
        act=act,
        bias_attr=biased,
        use_cudnn=False,
        name=name)
    return tmp


def atrous_conv(input,
                k_h,
                k_w,
                c_o,
                dilation,
                relu=False,
                padding="VALID",
                biased=False,
                name=None):
    act = None
    if relu:
        act = "relu"
    '''
    tmp = input
    if padding == "SAME":
        padding_h = max(k_h - s_h, 0)
        padding_w = max(k_w - s_w, 0)
        padding_top = padding_h // 2
        padding_left = padding_w // 2
        padding_bottom = padding_h - padding_top
        padding_right = padding_w - padding_left
        padding = [
            0, 0, 0, 0, padding_top, padding_bottom, padding_left, padding_right
        ]
        tmp = fluid.layers.pad(tmp, padding)
    '''
    tmp = fluid.layers.conv2d(
        input,
        num_filters=c_o,
        filter_size=[k_h, k_w],
        dilation=dilation,
        groups=1,
        act=act,
        bias_attr=biased,
        use_cudnn=False,
        name=name)
    return tmp


def zero_padding(input, padding):
    return fluid.layers.pad(input,
                            [0, 0, 0, 0, padding, padding, padding, padding])


def bn(input, relu=False, name=None, is_test=False):
    act = None
    if relu:
        act = 'relu'
    name = input.name.split(".")[0] + "_bn"
    tmp = fluid.layers.batch_norm(
        input, act=act, momentum=0.95, epsilon=1e-5, name=name)
    return tmp


def avg_pool(input, k_h, k_w, s_h, s_w, name=None, padding=0):
    temp = fluid.layers.pool2d(
        input,
        pool_size=[k_h, k_w],
        pool_type="avg",
        pool_stride=[s_h, s_w],
        pool_padding=padding,
        name=name)
    return temp


def max_pool(input, k_h, k_w, s_h, s_w, name=None, padding=0):
    temp = fluid.layers.pool2d(
        input,
        pool_size=[k_h, k_w],
        pool_type="max",
        pool_stride=[s_h, s_w],
        pool_padding=padding,
        name=name)
    return temp


def interp(input, out_shape):
    '''

    :param input: 图像张量
    :param out_shape: 输出大小
    :return: 双线性内插后缩放的大小
    '''
    out_shape = list(out_shape.astype("int32"))
    return fluid.layers.resize_bilinear(input, out_shape=out_shape)


def dilation_convs(input):
    tmp = res_block(input, filter_num=256, padding=1, name="conv3_2")
    tmp = res_block(tmp, filter_num=256, padding=1, name="conv3_3")
    tmp = res_block(tmp, filter_num=256, padding=1, name="conv3_4")

    tmp = proj_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_1")
    tmp = res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_2")
    tmp = res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_3")
    tmp = res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_4")
    tmp = res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_5")
    tmp = res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_6")

    tmp = proj_block(
        tmp, filter_num=1024, padding=4, dilation=4, name="conv5_1")
    tmp = res_block(tmp, filter_num=1024, padding=4, dilation=4, name="conv5_2")
    tmp = res_block(tmp, filter_num=1024, padding=4, dilation=4, name="conv5_3")
    return tmp


def pyramis_pooling(input, input_shape):
    shape = np.ceil(input_shape // 32).astype("int32")
    h, w = shape
    pool1 = avg_pool(input, h, w, h, w)
    pool1_interp = interp(pool1, shape)
    pool2 = avg_pool(input, h // 2, w // 2, h // 2, w // 2)
    pool2_interp = interp(pool2, shape)
    pool3 = avg_pool(input, h // 3, w // 3, h // 3, w // 3)
    pool3_interp = interp(pool3, shape)
    pool4 = avg_pool(input, h // 4, w // 4, h // 4, w // 4)
    pool4_interp = interp(pool4, shape)
    conv5_3_sum = input + pool4_interp + pool3_interp + pool2_interp + pool1_interp
    return conv5_3_sum


def shared_convs(image):
    print("shared_convs input", image.shape)
    tmp = conv(image, 3, 3, 32, 2, 2, padding='SAME', name="conv1_1_3_3_s2")
    tmp = bn(tmp, relu=True)
    tmp = conv(tmp, 3, 3, 32, 1, 1, padding='SAME', name="conv1_2_3_3")
    tmp = bn(tmp, relu=True)
    tmp = conv(tmp, 3, 3, 64, 1, 1, padding='SAME', name="conv1_3_3_3")
    tmp = bn(tmp, relu=True)
    tmp = max_pool(tmp, 3, 3, 2, 2, padding=[1, 1])

    tmp = proj_block(tmp, filter_num=128, padding=0, name="conv2_1")
    tmp = res_block(tmp, filter_num=128, padding=1, name="conv2_2")
    tmp = res_block(tmp, filter_num=128, padding=1, name="conv2_3")
    tmp = proj_block(tmp, filter_num=256, padding=1, stride=2, name="conv3_1")
    print("shared_convs input", tmp.shape)
    return tmp


def res_block(input, filter_num, padding=0, dilation=None, name=None):
    tmp = conv(input, 1, 1, filter_num // 4, 1, 1, name=name + "_1_1_reduce")
    tmp = bn(tmp, relu=True)
    tmp = zero_padding(tmp, padding=padding)
    if dilation is None:
        tmp = conv(tmp, 3, 3, filter_num // 4, 1, 1, name=name + "_3_3")
    else:
        tmp = atrous_conv(
            tmp, 3, 3, filter_num // 4, dilation, name=name + "_3_3")
    tmp = bn(tmp, relu=True)
    tmp = conv(tmp, 1, 1, filter_num, 1, 1, name=name + "_1_1_increase")
    tmp = bn(tmp, relu=False)
    tmp = input + tmp
    tmp = fluid.layers.relu(tmp)
    return tmp


def proj_block(input, filter_num, padding=0, dilation=None, stride=1,
               name=None):
    proj = conv(
        input, 1, 1, filter_num, stride, stride, name=name + "_1_1_proj")
    proj_bn = bn(proj, relu=False)

    tmp = conv(
        input, 1, 1, filter_num // 4, stride, stride, name=name + "_1_1_reduce")
    tmp = bn(tmp, relu=True)

    tmp = zero_padding(tmp, padding=padding)
    if padding == 0:
        padding = 'SAME'
    else:
        padding = 'VALID'
    if dilation is None:
        tmp = conv(
            tmp,
            3,
            3,
            filter_num // 4,
            1,
            1,
            padding=padding,
            name=name + "_3_3")
    else:
        tmp = atrous_conv(
            tmp,
            3,
            3,
            filter_num // 4,
            dilation,
            padding=padding,
            name=name + "_3_3")

    tmp = bn(tmp, relu=True)
    tmp = conv(tmp, 1, 1, filter_num, 1, 1, name=name + "_1_1_increase")
    tmp = bn(tmp, relu=False)
    tmp = proj_bn + tmp
    tmp = fluid.layers.relu(tmp)
    return tmp


def sub_net_4(input, input_shape):
    print("sub_net_4 input", input.shape)
    # !out_shape=(input_shape // 16)32更改为16
    tmp = interp(input, out_shape=(input_shape // 32))
    tmp = dilation_convs(tmp)
    tmp = pyramis_pooling(tmp, input_shape)
    tmp = conv(tmp, 1, 1, 256, 1, 1, name="conv5_4_k1")
    tmp = bn(tmp, relu=True)
    tmp = interp(tmp, out_shape=np.ceil(input_shape / 16))
    print("sub_net_4 out", tmp.shape)
    return tmp


def sub_net_2(input, input_shape):
    print("sub_net_2 input", input.shape)
    tmp = conv(input, 1, 1, 128, 1, 1, name="conv3_1_sub2_proj")
    tmp = bn(tmp, relu=False)
    # tmp = interp(tmp, out_shape=np.ceil(input_shape / 16))
    print("sub_net_2 out", tmp.shape)

    return tmp


def sub_net_1(input):
    print("sub_net_1 input", input.shape)
    tmp = conv(input, 3, 3, 32, 2, 2, padding='SAME', name="conv1_sub1")
    tmp = bn(tmp, relu=True)
    tmp = conv(tmp, 3, 3, 32, 2, 2, padding='SAME', name="conv2_sub1")
    tmp = bn(tmp, relu=True)
    tmp = conv(tmp, 3, 3, 64, 2, 2, padding='SAME', name="conv3_sub1")
    tmp = bn(tmp, relu=True)
    tmp = conv(tmp, 1, 1, 128, 1, 1, name="conv3_sub1_proj")
    tmp = bn(tmp, relu=False)
    print("sub_net_1 out", tmp.shape)
    return tmp


def CCF24(sub2_out, sub4_out, input_shape):
    print("CCF24 sub2_out", sub2_out.shape)
    print("CCF24 sub4_out", sub4_out.shape)
    tmp = zero_padding(sub4_out, padding=2)
    tmp = atrous_conv(tmp, 3, 3, 128, 2, name="conv_sub4")
    tmp = bn(tmp, relu=False)
    tmp = tmp + sub2_out
    tmp = fluid.layers.relu(tmp)
    tmp = interp(tmp, input_shape // 8)
    print("CCF24 out", tmp.shape)
    return tmp


def CCF124(sub1_out, sub24_out, input_shape):
    print("CCF124 sub1_out", sub1_out.shape)
    print("CCF124 sub24_out", sub24_out.shape)
    tmp = zero_padding(sub24_out, padding=2)
    tmp = atrous_conv(tmp, 3, 3, 128, 2, name="conv_sub2")
    tmp = bn(tmp, relu=False)
    tmp = tmp + sub1_out
    tmp = fluid.layers.relu(tmp)
    tmp = interp(tmp, input_shape // 4)
    return tmp


def icnet(data, num_classes, input_shape):
    '''

    :param data: 已经更改为0.8倍的图像数据
    :param num_classes: 分类数量
    :param input_shape: 输入图像数据的形状
    :return:
    '''
    image_sub1 = data
    image_sub2 = interp(data, out_shape=input_shape * 0.5)

    s_convs = shared_convs(image_sub2)  # 参数共享
    # s_convs=interp(s_convs,out_shape=s_convs.shape * 0.5)
    sub4_out = sub_net_4(s_convs, input_shape)
    sub2_out = sub_net_2(s_convs, input_shape)
    sub1_out = sub_net_1(image_sub1)

    sub24_out = CCF24(sub2_out, sub4_out, input_shape)
    sub124_out = CCF124(sub1_out, sub24_out, input_shape)

    conv6_cls = conv(
        sub124_out, 1, 1, num_classes, 1, 1, biased=True, name="conv6_cls")
    sub4_out = conv(
        sub4_out, 1, 1, num_classes, 1, 1, biased=True, name="sub4_out")
    sub24_out = conv(
        sub24_out, 1, 1, num_classes, 1, 1, biased=True, name="sub24_out")
    print('sub4_out', conv6_cls.shape, 'sub24_out', sub4_out.shape, 'sub124_out', sub124_out.shape)
    return sub4_out, sub24_out, conv6_cls
