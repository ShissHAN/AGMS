import tensorflow.compat.v1 as tf
from scipy.io import savemat
import numpy as np
tf.disable_v2_behavior()
import numpy as np
import numpy as np
import scipy.io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np

# 加载数据
data = scipy.io.loadmat('./data/AVIRIS.mat')
# 提取数据
prior_target = data['S']
prior_target = np.transpose(prior_target, (1, 0))
min_value = np.min(prior_target)
max_value = np.max(prior_target)
prior_target = (prior_target - min_value) / (max_value - min_value)
image = data['X']
[row,col,bands]=image.shape
image = np.reshape(image,[row*col,bands])

# 转换为TensorFlow张量
prior_target = tf.convert_to_tensor(prior_target)
image = tf.convert_to_tensor(image)
image = tf.cast(image, tf.float64)
# 计算 prior_target 的平均值
prior_target_mean = tf.reduce_mean(prior_target, axis=0, keepdims=True)
# 计算余弦相似度矩阵
similarity = tf.matmul(prior_target_mean, image, transpose_b=True)
similarity = tf.squeeze(similarity, axis=0)
# 取相似度最高的三个样本的索引
indices = tf.argsort(similarity, direction='DESCENDING')[:3]#ASCENDING升序，DESCENDING降序

# 使用tf.gather获取这三个样本
gathered_samples = tf.gather(image, indices)

# 使用tf.stack将这三个样本合并为一个张量
back_image = tf.stack(gathered_samples, axis=0)

with tf.Session() as sess:
    # 获取 TensorFlow 张量的值并转换为 NumPy 数组
    back_image = sess.run(back_image)
# 保存结果
scipy.io.savemat('./data/AVIRIS_bg.mat', {'similar_images': back_image})



def main():
    # 加载 .mat 文件
    data = scipy.io.loadmat('./data/AVIRIS_bg.mat')['similar_images']

    # 调用 Apply_PCA 函数
    transformed_data, num_components = Apply_PCA(data, fraction=15, choice=2)

    # 打印转换后的数据和保留的主成分数量
    print("Transformed Data:", transformed_data)
    print("Number of Components:", num_components)



def Apply_PCA(data, fraction, choice):
    """
    choice=1
    fraction：保留特征百分比
    choice=2
    fraction：降维后的个数
    """
    # PCA主成分分析
    if choice == 1:
        # 将数据转换为 TensorFlow 张量
        data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

        # 计算数据的协方差矩阵
        cov_matrix = tf.linalg.matmul(data_tensor, data_tensor, transpose_a=True) / tf.cast(tf.shape(data_tensor)[0],
                                                                                            tf.float32)

        # 计算特征向量和特征值
        eigenvalues, eigenvectors = tf.linalg.eigh(cov_matrix)

        # 保留指定百分比的特征值
        total_variance = tf.reduce_sum(eigenvalues)
        var_threshold = fraction * total_variance
        var_sum = tf.constant(0.0, dtype=tf.float32)
        num_components = 0
        for eigenvalue in eigenvalues[::-1]:
            var_sum += eigenvalue
            num_components += 1
            if var_sum >= var_threshold:
                break

        # 选择前 num_components 个特征向量
        selected_eigenvectors = eigenvectors[:, -num_components:]

        # 将数据转换到主成分空间
        img_pc = tf.linalg.matmul(data_tensor, selected_eigenvectors)

        # 将 TensorFlow 张量转换为 NumPy 数组
        img_pc = img_pc.numpy()

        print("PCA_DIM", num_components)  # 剩下的特征值数量

        return img_pc, num_components

    if choice == 2:
        # 将数据转换为 TensorFlow 张量
        data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

        # 将数据重新整形为二维数组
        #new_data = tf.reshape(data_tensor, (-1, tf.shape(data_tensor)[2]))
        new_data = data_tensor
        # 使用 tf.linalg.eigh 计算特征向量和特征值
        covariance_matrix = tf.matmul(tf.transpose(new_data), new_data)
        eigenvalues, eigenvectors = tf.linalg.eigh(covariance_matrix)

        # 选择前 fraction 个特征向量
        selected_eigenvectors = eigenvectors[:, -fraction:]

        # 将数据转换到主成分空间
        img_pc = tf.matmul(new_data, selected_eigenvectors)

        with tf.Session() as sess:
            # 获取 TensorFlow 张量的值并转换为 NumPy 数组
            img_pc = sess.run(img_pc)

        print("PCA_DIM", fraction)  # 剩下的特征值数量
        print(img_pc.shape)
        savemat('./data/AVIRIS_bg.mat', {'backnew': img_pc})
        return img_pc, fraction



if __name__ == "__main__":
    tf.disable_v2_behavior()
    main()


