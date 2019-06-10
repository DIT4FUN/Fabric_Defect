from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid


class FabricNet:
    '''
    纺织物瑕疵识别网络

    '''

    def __init__(self, class_dim=10):
        self.input = input
        self.class_dim = class_dim

    def interp(self, input, num):
        '''
        双线性内插
        :param input: 输入图像张量
        :param num: 缩放倍数
        :return: 返回双线性内插值
        '''
        h, w = input.shape[-2:]
        temp = fluid.layers.resize_bilinear(input, out_shape=[int(h * num), int(w * num)])
        return temp

    def conv_share(self, input):
        '''
        通用卷积函数
        :param input:
        :return:
        '''
        cp_1 = fluid.nets.simple_img_conv_pool(
            input=input,
            filter_size=3,  # 3x3滤波器
            num_filters=256,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=1,  # 填充
            act="relu")
        bn_1 = fluid.layers.batch_norm(cp_1)
        cp_2 = fluid.nets.simple_img_conv_pool(
            input=bn_1,
            filter_size=3,  # 3x3滤波器
            num_filters=256,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=1,  # 填充
            act="relu")
        bn_2 = fluid.layers.batch_norm(cp_2)
        cp_3 = fluid.nets.simple_img_conv_pool(
            input=bn_2,
            filter_size=3,  # 3x3滤波器
            num_filters=512,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=2,  # 填充
            act="relu")
        bn_3 = fluid.layers.batch_norm(cp_3)
        # cp_4 = fluid.nets.simple_img_conv_pool(
        #     input=bn_3,
        #     filter_size=3,  # 3x3滤波器
        #     num_filters=512,  # 卷积核数量
        #     pool_size=2,  # 池化x2
        #     pool_stride=2,  # 池化步长2
        #     conv_padding=2,  # 填充
        #     act="relu")
        # bn_4 = fluid.layers.batch_norm(cp_4)
        # cp_5 = fluid.nets.simple_img_conv_pool(
        #     input=bn_4,
        #     filter_size=3,  # 3x3滤波器
        #     num_filters=1024,  # 卷积核数量
        #     pool_size=2,  # 池化x2
        #     pool_stride=2,  # 池化步长2
        #     conv_padding=4,  # 填充
        #     act="relu")
        # bn_5 = fluid.layers.batch_norm(cp_5)
        return bn_3

    def net(self, input):
        path1_input = self.interp(input, 0.065)
        path2_input = self.interp(input, 0.3)
        path3_input = self.interp(input, 0.8)
        path1_conv = self.conv_share(path1_input)
        path2_conv = self.conv_share(path2_input)
        # path2
        p2_cp_4 = fluid.nets.simple_img_conv_pool(
            input=path2_conv,
            filter_size=3,  # 3x3滤波器
            num_filters=512,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=2,  # 填充
            act="relu")
        p2_bn_4 = fluid.layers.batch_norm(p2_cp_4)
        p2_cp_5 = fluid.nets.simple_img_conv_pool(
            input=p2_bn_4,
            filter_size=3,  # 3x3滤波器
            num_filters=1024,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=2,  # 填充
            act="relu")
        p2_bn_5 = fluid.layers.batch_norm(p2_cp_5)
        # path3
        path3_conv = self.conv_share(path3_input)
        p3_cp_4 = fluid.nets.simple_img_conv_pool(
            input=path3_conv,
            filter_size=3,  # 3x3滤波器
            num_filters=512,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=2,  # 填充
            act="relu")
        p3_bn_4 = fluid.layers.batch_norm(p3_cp_4)


        # concat = fluid.layers.concat(
        #    [path1_conv, path2_conv, path3_conv], axis=1)
        print("path1", path1_conv.shape)
        print("path2", p2_bn_5.shape)
        print("path3", p3_bn_4.shape)
        return 0
