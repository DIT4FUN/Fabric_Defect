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
        cp_1 = fluid.layers.conv2d(
            input=input,
            num_filters=32,
            filter_size=3,
            act="relu",
            padding=1
        )
        bn_1 = fluid.layers.batch_norm(cp_1)
        # cp_2 = fluid.nets.simple_img_conv_pool(
        #     input=bn_1,
        #     filter_size=3,  # 3x3滤波器
        #     num_filters=256,  # 卷积核数量
        #     pool_size=2,  # 池化x2
        #     pool_stride=2,  # 池化步长2
        #     conv_padding=1,  # 填充
        #     act="relu")
        # bn_2 = fluid.layers.batch_norm(cp_2)
        # cp_3 = fluid.nets.simple_img_conv_pool(
        #     input=bn_2,
        #     filter_size=3,  # 3x3滤波器
        #     num_filters=512,  # 卷积核数量
        #     pool_size=2,  # 池化x2
        #     pool_stride=2,  # 池化步长2
        #     conv_padding=2,  # 填充
        #     act="relu")
        # bn_3 = fluid.layers.batch_norm(cp_3)

        return bn_1

    def net(self, input):
        path1_input = self.interp(input, 0.075)  # x1
        path2_input = self.interp(input, 0.3)  # x4
        path3_input = self.interp(input, 0.6)  # x8
        path1_conv = self.conv_share(path1_input)
        path2_conv = self.conv_share(path2_input)
        # path2
        p2_cp_4 = fluid.nets.simple_img_conv_pool(
            input=path2_conv,
            filter_size=3,  # 3x3滤波器
            num_filters=64,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=1,  # 填充
            act="relu")
        p2_bn_4 = fluid.layers.batch_norm(p2_cp_4)

        # path3
        path3_conv = self.conv_share(path3_input)
        p3_cp_4 = fluid.nets.simple_img_conv_pool(
            input=path3_conv,
            filter_size=3,  # 3x3滤波器
            num_filters=64,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=1,  # 填充
            act="relu")
        p3_bn_4 = fluid.layers.batch_norm(p3_cp_4)
        p3_cp_5 = fluid.nets.simple_img_conv_pool(
            input=p3_bn_4,
            filter_size=3,  # 3x3滤波器
            num_filters=64,  # 卷积核数量
            pool_size=2,  # 池化x2
            pool_stride=2,  # 池化步长2
            conv_padding=1,
            act="relu")

        p3_bn_5 = fluid.layers.batch_norm(p3_cp_5)

        p3_final = p3_bn_5
        p2_final = p2_bn_4
        p1_final = fluid.layers.resize_bilinear(path1_conv, out_shape=p2_bn_4.shape[-2:])
        print("path1", p1_final.shape)
        print("path2", p2_final.shape)
        print("path3", p3_final.shape)

        # 需要改
        out0 = fluid.layers.concat(
            [p1_final, p2_final, p3_final], axis=1)
        out = fluid.layers.conv2d(out0, 128, 1, 1)
        out = fluid.layers.conv2d(out, 64, 1, 1)
        out = fluid.layers.conv2d(out, self.class_dim, 1, 1)
        out_final = fluid.layers.resize_bilinear(out, out_shape=[i//4 for i in input.shape[-2:]])
        print("out0", out0.shape)
        print("out_F", out_final.shape)
        return out_final
