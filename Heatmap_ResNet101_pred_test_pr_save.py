# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
import tensorflow.keras as keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='FangSong')
import numpy as np

# tensorflow版本
print("tf.version:", tf.__version__)

# 获取类名称
# train_path = r'D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\train'
# cwd = train_path + '\\'
# className = os.listdir(train_path)
# for i in range(len(className)):
#     c = re.split("_", className[i])
#     className[i] = c[1]+"_"+c[2]
# print("64个类：", className)
className = ['1708_CM2', '1708_CM', '1736_Y', '1757_4GSY', '1757_4G', '1757_6G', '1757_8GSY', '1757_8G', '1757_BSY',
             '1757_B', '1757_CM2', '1757_CM', '1757_Y', '1757_橱柜门', '1757_面板', '1757_面板2', '1765_4GSY', '1765_4G',
             '1765_6G', '1765_8GSY', '1765_8G', '1765_BSY', '1765_B', '1765_CM2', '1765_CM', '1765_Y', '1765_橱柜门',
             '1765_面板', '1765_面板2', '1770_4GSY', '1770_4G', '1770_6GSY', '1770_6G', '1770_8GSY', '1770_8G', '1770_BSY',
             '1770_B', '1770_CM2', '1770_CM', '1770_Y', '1770_橱柜门', '1770_面板', '1770_面板2', '1771_4GSY', '1771_4G',
             '1771_6G', '1771_8GSY', '1771_8G', '1771_BSY', '1771_B', '1771_CM2', '1771_CM', '1771_Y', '1771_橱柜门',
             '1771_面板', '1771_面板2', '1773_Y', '1782_Y', '1783_B', '1783_Y', '1783_橱柜门', '1783_面板', '1786_Y', '1787_Y']
# 从tfrecord得到数据集
train_tfrecords_file = r'D:\MyFiles\ResearchSubject\Alldatasets3\tfrecords\train.tfrecords'
valid_tfrecords_file = r'D:\MyFiles\ResearchSubject\Alldatasets3\tfrecords\valid.tfrecords'
test_tfrecords_file = r'D:\MyFiles\ResearchSubject\Alldatasets3\tfrecords\test.tfrecords'
train_dataset = tf.data.TFRecordDataset(train_tfrecords_file)
valid_dataset = tf.data.TFRecordDataset(valid_tfrecords_file)
test_dataset = tf.data.TFRecordDataset(test_tfrecords_file)

features = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'img_raw': tf.io.FixedLenFeature([], tf.string)
}

def read_and_decode(example_string):
    features_dic = tf.io.parse_single_example(example_string, features)  # 解析example序列变成的字符串序列
    img = tf.io.decode_raw(features_dic['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    label = tf.cast(features_dic['label'], tf.int32)
    return img, label

from MeanStd import MeanStd
mean, std = MeanStd().Getmean_std()

def standardize(image_data):
    image_data = tf.cast(image_data, tf.float32)
    image_data = (image_data - mean)/std
    image_data = tf.reverse(image_data, [-1])
    # 将RGB转成BGR，符合VGG16预训练模型的输入要求（预处理要求）
    # 在VGG16中预处理要求还有一条要进行中心化，但是如果采用VGG16默认的预处理方法，则中心化是以ImageNet数据集而言的，因此不能采用VGG16
    # 默认的预处理方法
    return image_data

def getBatchDataset(dataset, batch=64):
    dataset = dataset.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().repeat(1).shuffle(buffer_size=32000)
    dataset = dataset.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=batch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def getBatchDataset2(dataset, batch=64):
    dataset = dataset.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(1).shuffle(buffer_size=32000, seed=12, reshuffle_each_iteration=False)
    dataset = dataset.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=batch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
train_batch_dataset = getBatchDataset(train_dataset, TRAIN_BATCH_SIZE)
valid_batch_dataset = getBatchDataset2(valid_dataset, VALID_BATCH_SIZE)
test_batch_dataset = getBatchDataset2(test_dataset, TEST_BATCH_SIZE)

def reverse_standardize(image_data):
    image_data = tf.reverse(image_data, [-1])
    image_data = np.clip(image_data * std + mean, 0, 255)
    return image_data

def displayImages(dataset):
    plt.figure(figsize=(10, 10))
    # 整个画布（包括各子图在内）的大小是1000×1000
    images, labels = next(iter(dataset))
    # print(labels)
    # 取一个batch的数据
    for i in range(9):
        # img = tf.squeeze(imags[i], 2)
        plt.subplot(3, 3, i + 1)
        plt.imshow(reverse_standardize(images[i]).astype('uint8'))
        plt.title(className[tf.cast(labels[i], tf.int32)])
        plt.axis('off')
    plt.show()
#
# displayImages(train_batch_dataset)
# displayImages(valid_batch_dataset)

# 创建模型
inputs = keras.Input(shape=(256, 256, 3), name="my_input")
from Heatmap_ResNet101_pred import ClassModel
model = ClassModel(inputs).CreateMyModel()


# 加载上次结束训练时的权重
model.load_weights(r"doorModels\cp-054-0.039-1.000-0.167-0.962.h5", by_name=True)
print('successfully loading weights')
# model.summary()

myDataset = test_batch_dataset
myPath = r"numpy\test"
os.makedirs(myPath, exist_ok=True)

predPath = myPath+r'\pred.npy'
onehot_labelPath = myPath+r'\onehot_label.npy'
prPath = myPath+r'\pr.npy'

final1 = None
final2 = None
final = None
i = 0

for images, labels in myDataset:
    print("batch:", i+1)
    batch_value = images.shape[0]
    print("batch_value:", batch_value)
    if i == 0:
        final1 = model.predict(images)
        final2 = np.eye(64)[np.array(labels)]

        labels = np.array(labels).reshape(batch_value, 1)
        final = np.concatenate((labels, final1), axis=1)

    else:
        pred = model.predict(images)
        labeli = np.eye(64)[np.array(labels)]
        # print(predList, labels)
        final1 = np.concatenate((final1, pred), axis=0)
        final2 = np.concatenate((final2, labeli), axis=0)

        labels = np.array(labels).reshape(batch_value, 1)
        finali = np.concatenate((labels, pred), axis=1)
        final = np.concatenate((final, finali), axis=0)
    i += 1

np.save(predPath, final1)
np.save(onehot_labelPath, final2)
np.save(prPath, final)
