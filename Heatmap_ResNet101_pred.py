"""
预测时的模型
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import matplotlib
matplotlib.rc("font", family='FangSong')
import tensorflow as tf

class ClassModel():
    def __init__(self, inputs1):
        self.images = inputs1

    def CreateMyModel(self):
        # 第一个网络
        base_model = keras.applications.ResNet101(input_tensor=self.images, include_top=False, weights='imagenet')
        # base_model.trainable = False
        fc1_4 = keras.layers.Dense(64, name='out1_score')
        out1_output = keras.layers.Activation('softmax', name='out1')
        x = base_model.output
        x = keras.layers.GlobalAvgPool2D(name='conv5_block3_out_GAP')(x)
        out1_score = fc1_4(x)
        out1 = out1_output(out1_score)
        self.model1 = model1 = keras.models.Model(self.images, out1)
        model1.trainable = False
        # model1.load_weights("doorModels\cp-030-0.014-0.995-1.000-0.995-0.261-0.941-0.953-0.939.h5")
        # model1.summary()

        # 第二个网络
        features = model1.layers[-4].output
        size = (256, 256, 3)
        newImage, heatmap = keras.layers.Lambda(self.getNewImage, name='NewImage')([self.images, out1_score, features, size])

        base_model2 = keras.applications.ResNet101(input_shape=size, include_top=False, weights=None, pooling='avg')
        # base_model2.load_weights(r"doorModels\cp-030-0.014-0.995-1.000-0.995-0.261-0.941-0.953-0.939.h5",
        #                          by_name=True)
        base_model2.trainable = False
        x = base_model2(newImage)
        fc2_4 = keras.layers.Dense(64, name='out2_score')
        out2_output = keras.layers.Activation('softmax', name='out2')
        out2_score = fc2_4(x)
        out2 = out2_output(out2_score)

        out3 = keras.layers.Average(name='out3')([out1, out2])
        model = keras.models.Model(inputs=model1.input, outputs=out3)
        return model

    def getNewImage(self, X_score_features_size):
        threashold = 0.2
        X = X_score_features_size[0]
        # print(X)
        B = tf.shape(X)[0]
        # tf.print(B)
        # print(B)

        score = X_score_features_size[1]
        features = X_score_features_size[2]
        # tf.print('score:', score)
        # tf.print('features:', features)
        # print(features.get_shape()[1:3])
        # print(tf.shape(features))
        size = X_score_features_size[3][0]
        # labels = X_score_features_size[4]
        # tf.print("labels:", labels)
        # print("labels:", labels)
        out = tf.zeros((B, )+X_score_features_size[3])
        heatmap = tf.zeros((B, )+(8, 8))
        # print(heatmap)
        if B != None:
            # 预测时使用预测所在的类
            c1 = tf.concat((tf.reshape(tf.range(0, B), [-1, 1]),
                           tf.cast(tf.reshape(tf.argmax(score, axis=1), [-1, 1]), tf.int32)), axis=1)
            # c2 = tf.concat((tf.reshape(tf.range(0, B), [-1, 1]),
            #                 tf.cast(tf.reshape(labels, [-1, 1]), tf.int32)), axis=1)
            score = tf.gather_nd(score, c1)
            gradient = tf.gradients(score, features)[0]
            # print("gradient:", gradient)
            # tf.print("gradient:", gradient)
            if gradient != None:
                # 实验发现GMP效果比GAP好
                pooled_grads = tf.reshape(keras.backend.max(gradient, axis=(1, 2)), [B, 1, 1, 2048])
                # pooled_grads2 = tf.reshape(keras.backend.mean(gradient, axis=(1, 2)), [B, 1, 1, 512])
                # pooled_grads = tf.add(pooled_grads, pooled_grads2)
                heatmap = tf.reduce_sum(tf.multiply(pooled_grads, features), axis=-1)
                # 权重与卷积层输出相乘，按通道层求和或平均（求和或者求平均都可以不影响）
                heatmap = tf.maximum(heatmap, 0)  # 取heatmap和0中的最大值，即relu激活
                #     # print("heatmap:", heatmap)
                max_heat = tf.reduce_max(tf.reduce_max(heatmap, axis=-1), axis=-1)
                #     # print("max_heat：", max_heat)
                max_heat = tf.where(max_heat == 0, 1e-10, max_heat)
                b2 = tf.reshape(max_heat, [B, 1, 1])
                heatmap = tf.divide(heatmap, b2)  # 其实就是(heatmap-min_heat) / (max_heat - min_heat)，这里min_heat==0
                # 这里就是将heatmap每个位置的值归一化到0~1
                # tf.print("heatmap:", heatmap)
                # 二值化，自定义灰度界限，大于这个值为黑色，小于这个值为白色
                heatmap = tf.where(heatmap > threashold, 255, 0)
                indice = tf.where(heatmap == 255)
                # tf.print("indice:", indice)
                # c = tf.range(0, B)
                start = tf.reduce_min(tf.where(indice[:, 0] == 0))
                stop = tf.reduce_max(tf.where(indice[:, 0] == 0))
                indice3 = indice[start:stop + 1, :]
                # print("indice3:", indice3)
                # tf.print("indice3:", indice3)
                indice4 = indice3[:, 1]
                min_x = tf.reduce_min(indice4)
                # print("min_x:", min_x)
                max_x = tf.reduce_max(indice4)
                indice5 = indice3[:, 2]
                min_y = tf.reduce_min(indice5)
                max_y = tf.reduce_max(indice5)
                a = tf.stack((min_x, max_x, min_y, max_y), axis=0)
                i = tf.constant(1, dtype=tf.int32)
                loop_cond = lambda x, y, z: tf.less(y, B)
                loop_vars = [indice, i, a]
                indice, i, a = tf.while_loop(loop_cond, self.getCoordinates, loop_vars,
                                             shape_invariants=[indice.get_shape(), i.get_shape(),
                                                               tf.TensorShape([None, ])])
                coordinates = tf.reshape(a, [B, 4])
                # tf.print("coordinates:", coordinates)
                out = self.Zoom(X, coordinates, size)
                # tf.print("out:", out)
        return out, heatmap

    def getCoordinates(self, indice, i, a):
        start = tf.reduce_min(tf.where(indice[:, 0] == tf.cast(i, tf.int64)))
        stop = tf.reduce_max(tf.where(indice[:, 0] == tf.cast(i, tf.int64)))
        indice3 = indice[start:stop + 1, :]
        # print("indice3:", indice3)
        indice4 = indice3[:, 1]
        min_x = tf.reduce_min(indice4)
        max_x = tf.reduce_max(indice4)
        indice5 = indice3[:, 2]
        min_y = tf.reduce_min(indice5)
        max_y = tf.reduce_max(indice5)
        b = tf.stack((min_x, max_x, min_y, max_y), axis=0)
        a = tf.concat((a, b), axis=0)
        i = tf.add(i, 1)
        return indice, i, a

    def get_pixel_value(self, img, x, y):
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, x, y], 3)
        out = tf.gather_nd(img, indices)
        return tf.cast(out, tf.float32)

    def Zoom(self, X, coordinates, size):
        batch = tf.shape(X)[0]

        ltx = tf.expand_dims(tf.reshape(coordinates[:, 0], [-1, 1]), -1)
        lty = tf.expand_dims(tf.reshape(coordinates[:, 2], [-1, 1]), -1)
        rbx = tf.expand_dims(tf.reshape(coordinates[:, 1], [-1, 1]), -1)
        rby = tf.expand_dims(tf.reshape(coordinates[:, 3], [-1, 1]), -1)


        zero = tf.zeros([], dtype='float32')
        one = tf.ones([], dtype='float32')

        out = tf.zeros_like(size)

        ltx = ltx * 255 / 7
        lty = lty * 255 / 7
        rbx = rbx * 255 / 7
        rby = rby * 255 / 7

        # ltx = tf.constant([[72.]])  # 竖直方向是x, 水平方向是y
        # lty = tf.constant([[72.]])
        # rbx = tf.constant([[183.]])
        # rby = tf.constant([[183.]])

        ltx = tf.cast(ltx, tf.float32)
        lty = tf.cast(lty, tf.float32)
        rbx = tf.cast(rbx, tf.float32)
        rby = tf.cast(rby, tf.float32)

        # tf.print("ltx:{}, lty:{}, rbx:{}, rby:{}".format(ltx, lty, rbx, rby))

        if batch != None:
            # 目标图像坐标
            x_target = tf.tile(tf.reshape(tf.range(0, size), [1, -1, 1]), [batch, 1, size])
            x_target = tf.cast(x_target, tf.float32)
            y_target = tf.tile(tf.reshape(tf.range(0, size), [1, 1, -1]), [batch, size, 1])
            y_target = tf.cast(y_target, tf.float32)


            #     # 原图像浮点数坐标
            x2_0 = x_target * (rbx - ltx + 1) / tf.cast(size, tf.float32)
            y2_0 = y_target * (rby - lty + 1) / tf.cast(size, tf.float32)
            # print(x2_0, y2_0)
            #
            #     # 原图像最靠近浮点数坐标的4个整数坐标
            x2_source = tf.math.floordiv(x_target * (rbx - ltx + 1), tf.cast(size, tf.float32))
            y2_source = tf.math.floordiv(y_target * (rby - lty + 1), tf.cast(size, tf.float32))
            x2_source_1 = x2_source + 1
            y2_source_1 = y2_source + 1

            # 前景对象区域限制（相对于前景对象区域而言，而不是原图像）
            x2_source = tf.clip_by_value(x2_source, zero, tf.round(rbx - ltx)-1)
            x2_source_1 = tf.clip_by_value(x2_source_1, one, tf.round(rbx - ltx))
            y2_source = tf.clip_by_value(y2_source, zero, tf.round(rby - lty)-1)
            y2_source_1 = tf.clip_by_value(y2_source_1, one, tf.round(rby - lty))
            # print("x2_source", x2_source)

            # 从原图像取坐标时，要偏移前景对象区域的左上角坐标
            Ia = self.get_pixel_value(X, tf.cast(x2_source + ltx, tf.int32),
                                 tf.cast(y2_source + lty, tf.int32))
            # print("Ia:", Ia)
            Ib = self.get_pixel_value(X, tf.cast(x2_source + ltx, tf.int32),
                                 tf.cast(y2_source_1 + lty, tf.int32))
            Ic = self.get_pixel_value(X, tf.cast(x2_source_1 + ltx, tf.int32),
                                 tf.cast(y2_source + lty, tf.int32))
            Id = self.get_pixel_value(X, tf.cast(x2_source_1 + ltx, tf.int32),
                                 tf.cast(y2_source_1 + lty, tf.int32))

            # 原图像最靠近浮点数坐标的4个整数坐标的系数值
            wa = (x2_source_1 - x2_0) * (y2_source_1 - y2_0)
            wb = (x2_source_1 - x2_0) * (y2_0 - y2_source)
            wc = (x2_0 - x2_source) * (y2_source_1 - y2_0)
            wd = (x2_0 - x2_source) * (y2_0 - y2_source)

            # add dimension for addition
            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)
            # tf.print("X_new", X_new)

            # 双线性插值
            out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            # print("out:", out)

        return out


    def Zoom2(self, X, coordinates, size):
        batch = tf.shape(X)[0]
        # Height = tf.shape(X)[1]
        Width = tf.shape(X)[2]

        ltx = tf.expand_dims(tf.reshape(coordinates[:, 0], [-1, 1]), -1)
        lty = tf.expand_dims(tf.reshape(coordinates[:, 2], [-1, 1]), -1)
        rbx = tf.expand_dims(tf.reshape(coordinates[:, 1], [-1, 1]), -1)
        rby = tf.expand_dims(tf.reshape(coordinates[:, 3], [-1, 1]), -1)

        # tf.print("ltx:{}, lty:{}, rbx:{}, rby:{}".format(ltx, lty, rbx, rby))

        zero = tf.zeros([], dtype='float32')
        one = tf.ones([], dtype='float32')
        # source_max_x_y = tf.cast(Width - 1, 'float32')
        # feature_max_x_y = tf.cast(Width - 1, 'float32')

        out = tf.zeros_like(size)

        ltx = ltx * 255 / 7
        lty = lty * 255 / 7
        rbx = rbx * 255 / 7
        rby = rby * 255 / 7

        ltx = tf.cast(ltx, tf.float32)
        lty = tf.cast(lty, tf.float32)
        rbx = tf.cast(rbx, tf.float32)
        rby = tf.cast(rby, tf.float32)

        tf.print(ltx)

        if batch != None:
            # 目标图像坐标
            x_target = tf.tile(tf.reshape(tf.range(0, Width), [1, -1, 1]), [batch, 1, Width])
            x_target = tf.cast(x_target, tf.float32)
            y_target = tf.tile(tf.reshape(tf.range(0, Width), [1, 1, -1]), [batch, Width, 1])
            y_target = tf.cast(y_target, tf.float32)
            M = (self.h(x_target - ltx) - self.h(x_target - rbx)) * (self.h(y_target - lty) - self.h(y_target - rby))
            M = tf.tile(tf.expand_dims(M, -1), tf.constant([1, 1, 1, 3], tf.int32))

            out = tf.multiply(X, M)
            out = tf.image.resize_with_pad(out, size, size, antialias=True)
        return out


    def h(self, x):
        k = 100
        h = 1 / (1 + tf.math.exp(-k * x))
        # print(h)
        # h = tf.where(x > 0, 1., 0.)
        return h
#
# inputs = keras.Input(shape=(256, 256, 3), name="images")
# b = ClassModel(inputs).CreateMyModel()
# b.summary()

