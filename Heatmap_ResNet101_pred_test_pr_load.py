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
    image_data = (image_data - mean) / std
    image_data = tf.reverse(image_data, [-1])
    # 将RGB转成BGR，符合VGG16预训练模型的输入要求（预处理要求）
    # 在VGG16中预处理要求还有一条要进行中心化，但是如果采用VGG16默认的预处理方法，则中心化是以ImageNet数据集而言的，因此不能采用VGG16
    # 默认的预处理方法
    return image_data


def getBatchDataset(dataset, batch=64):
    dataset = dataset.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().repeat(1).shuffle(
        buffer_size=32000)
    dataset = dataset.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        batch_size=batch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def getBatchDataset2(dataset, batch=64):
    dataset = dataset.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(1).shuffle(
        buffer_size=32000, seed=12, reshuffle_each_iteration=False)
    dataset = dataset.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        batch_size=batch)
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

# 绘制pr曲线
myPath = r"\test"
prPath = r"numpy" + myPath + r'\pr.npy'
pr = np.load(prPath)
all_labels = pr[:, 0]
#
# for i in range(64):
#     print(np.where(all_labels == i))
#     a = np.sort(np.unique(pr[:, 1+i]))
#     print(a.shape)

# np.cumsum()

prob_all = np.sort(np.unique(pr[:, 1::]))
print(prob_all, prob_all.shape)

macro_precision = []
macro_recall = []

# print(dic)
prob_all = np.linspace(0, 1, 11)
# prob_all = np.array([0., 0.1, 0.5, 1.0])

#
for prob_index in range(prob_all.shape[0]):
    dic = {}
    for i in range(64):
        dic['tp' + str(i)] = 0
        dic['fn' + str(i)] = 0
        dic['fp' + str(i)] = 0
    for j in range(pr.shape[0]):
        label = pr[j, 0]
        # print(label)
        pred_prob = pr[j, 1::]
        # print(pred_prob)
        for cls in range(64):
            # print(label, cls)
            # print(pred_prob[cls], prob_index)
            if pred_prob[cls] >= prob_all[prob_index] and label == cls:
                dic['tp' + str(cls)] += 1
            elif pred_prob[cls] >= prob_all[prob_index] and label != cls:
                dic['fp' + str(cls)] += 1
            elif pred_prob[cls] < prob_all[prob_index] and label == cls:
                dic['fn' + str(cls)] += 1
    print(dic)
    precision = 0
    recall = 0
    for index in range(64):
        if index == 0:
            if dic['tp' + str(index)] == 0 and dic['fp' + str(index)] == 0:
                precision = 1
            else:
                precision = dic['tp' + str(index)] / (dic['tp' + str(index)] + dic['fp' + str(index)] + 1e-10)
            recall = dic['tp' + str(index)] / (dic['tp' + str(index)] + dic['fn' + str(index)] + 1e-10)
        else:
            if dic['tp' + str(index)] == 0 and dic['fp' + str(index)] == 0:
                precisioni = 1
            else:
                precisioni = dic['tp' + str(index)] / (dic['tp' + str(index)] + dic['fp' + str(index)] + 1e-10)
            recalli = dic['tp' + str(index)] / (dic['tp' + str(index)] + dic['fn' + str(index)] + 1e-10)
            precision += precisioni
            recall += recalli
    precision = precision / 64
    recall = recall / 64
    macro_precision.append(precision)
    macro_recall.append(recall)

print("macro_recall:", macro_recall)
print("macro_precision:", macro_precision)

# x = np.array(macro_recall)
# y = np.array(macro_precision)
plt.figure()
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR curve')
plt.plot(macro_recall, macro_precision)
plt.show()
plt.savefig('pr_curve.png')


# 计算mAP
# 使用各阶段的每个precision*recall变化，而不是使用各阶段最大的precision*recall的变化
myPath = r"\test"
predPath = r"numpy"+myPath+r'\pred.npy'
onehot_labelPath = r"numpy"+myPath+r'\onehot_label.npy'
preds = np.load(predPath)
targs = np.load(onehot_labelPath)


def average_precision(output, target):
    epsilon = 1e-8
    indices = output.argsort()[::-1]
    total_count_ = np.cumsum(np.ones((len(output), 1)))
    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]

    # recall = pos_count_ / total
    # print("recall:", recall)
    # myPrecision = pos_count_ / total_count_
    # print("myPrecision:", myPrecision)
    # print(np.sum(myPrecision))
    # final = recall * myPrecision
    # print("final:", final)
    # final = final.sum()
    # print("final:", final)


    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)
    return precision_at_i

def mAP(targs, preds):
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))

    for k in range(preds.shape[1]):
        output = preds[:, k]
        target = targs[:, k]
        ap[k] = average_precision(output, target)
    print("各类的average precision值：", ap)
    return ap.mean()

print("最终的mean Average Precision值：", mAP(targs, preds))

