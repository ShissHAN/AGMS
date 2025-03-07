'''
根据生成的目标数据集，使用PCA进行降维处理
'''
import tensorflow.compat.v1 as tf
from scipy.io import savemat
tf.disable_v2_behavior()
import scipy.io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def main():
    # 加载 .mat 文件
    data = scipy.io.loadmat('./gen_target_AV.mat')['gen_target_AV']
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
        savemat('./data/gen_target_15.mat', {'target_15': img_pc})
        return img_pc, fraction



if __name__ == "__main__":
    tf.disable_v2_behavior()
    main()


