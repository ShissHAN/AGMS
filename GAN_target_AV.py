import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat
from scipy.io import savemat
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
z_real = loadmat('./data/AVIRIS_bg.mat')
hyp_img = loadmat('./data/AVIRIS.mat')
data = hyp_img['S']
data = np.transpose(data, (1, 0))
min_value = np.min(data)
max_value = np.max(data)
data = (data - min_value) / (max_value - min_value)
[num, bands]=data.shape
d_num=1
num_examples = num
input_dim = bands
n_l1 = 500
n_l2 = 500
z_dim = 15
batch_size = num
n_epochs = 100
learning_rate = 1e-3
learning_rate_discriminator = 1e-4
beta1 = 0.8
results_path = './resultAV/'

x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')
x_vector = tf.placeholder(dtype=tf.float32, shape=[1, input_dim], name='xvector')
decoder_output_vector = tf.placeholder(dtype=tf.float32, shape=[1, input_dim], name='decodervector')

def form_results():
    saved_model_path = results_path  + '/Saved_models/'
    log_path = results_path  + '/log'
    encoder_path = results_path  + '/encoder/'
    decoder_path = results_path  + '/decoder/'
    if not os.path.exists(results_path ):
        os.mkdir(results_path )
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
        os.mkdir(encoder_path)
        os.mkdir(decoder_path)
    return saved_model_path, log_path, encoder_path, decoder_path

def LeakyRelu(x,leaky=0.2,name = 'LeakyRelu'):
    with tf.variable_scope(name, reuse=None):
        f1 = 0.5 * (1 + leaky)
        f2 = 0.5 * (1 - leaky)
        return f1 * x + f2 * tf.abs(x)

def norm_BN(x, n1, n2, name, reuse=False):
     if reuse:       
         tf.get_variable_scope().reuse_variables()
     with tf.name_scope('BN'):
         mean, var = tf.nn.moments(x, axes=[0])
         scale = tf.Variable(tf.ones([n2]))
         shift = tf.Variable(tf.zeros([n2]))
         epsilon = 0.00000001
         x_bn = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
     return x_bn

def dense(x, n1, n2, name):
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0.,
                                                                           stddev=0.01))

        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out

def encoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_temp_1 = dense(x, input_dim, n_l1, 'e_dense_1')
        e_temp_1_bn = norm_BN(e_temp_1, batch_size, n_l1, 'e_bn_1')
        e_dense_1 = LeakyRelu(e_temp_1_bn)
        e_temp_2 = dense(e_dense_1, n_l1, n_l2, 'e_dense_2')
        e_temp_2_bn = norm_BN(e_temp_2, batch_size, n_l2, 'e_bn_2')
        e_dense_2 = LeakyRelu(e_temp_2_bn)
        e_temp_3 = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
        latent_variable = e_temp_3
        return latent_variable

def decoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_temp_1 = dense(x, z_dim, n_l2, 'd_dense_1')
        d_temp_1_bn = norm_BN(d_temp_1, batch_size , n_l1, 'd_bn_1')
        d_dense_1 = LeakyRelu(d_temp_1_bn)
        d_temp_2 = dense(d_dense_1, n_l2, n_l1, 'd_dense_2')
        d_temp_2_bn = norm_BN(d_temp_2, batch_size , n_l2, 'd_bn_2')
        d_dense_2 = LeakyRelu(d_temp_2_bn)
        d_temp_3 = dense(d_dense_2, n_l1, input_dim, 'd_output')
        output = tf.nn.tanh(d_temp_3)
        return output


def discriminator(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator'):
        out1 = dense(x, z_dim, n_l1, name='dc_den1')
        dc_temp_1_bn = norm_BN(out1, batch_size , n_l1, 'dc_bn1')
        dc_den1 = LeakyRelu(dc_temp_1_bn)
        out2 = dense(dc_den1, n_l1, n_l2, name='dc_den2')
        dc_temp_2_bn = norm_BN(out2, batch_size , n_l2, 'dc_bn2')
        dc_den2 = LeakyRelu(dc_temp_2_bn)
        out3 = dense(dc_den2, n_l2, 1, name='dc_output')
        output =  tf.nn.sigmoid(out3)#n_l1
        return output

def adversary(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('adversary'):
        out1 = dense(x, input_dim, n_l1, name='ad_den1')
        ad_temp_1_bn = norm_BN(out1, batch_size , n_l1, 'ad_bn1')
        ad_den1 = LeakyRelu(ad_temp_1_bn)
        out2 = dense(ad_den1, n_l1, n_l2, name='ad_den2')
        ad_temp_2_bn = norm_BN(out2, batch_size , n_l2, 'ad_bn2')
        ad_den2 = LeakyRelu(ad_temp_2_bn)
        out3 = dense(ad_den2, n_l2, 1, name='ad_output')
        output =  tf.nn.sigmoid(out3)
        return output
#训练
def train(train_model=True):
    
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder(x_input)
        decoder_output = decoder(encoder_output)
        
    with tf.variable_scope(tf.get_variable_scope()):
        d_real = discriminator(real_distribution)
        d_fake = discriminator(encoder_output, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        x_real = adversary(x_input)
        x_fake = adversary(decoder_output, reuse=True)    

    autoencoder_loss = tf.reduce_mean(tf.square(x_input - decoder_output))
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = 0.5*(dc_loss_fake + dc_loss_real)

    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    adversary_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(x_real), logits=x_real))
    adversary_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(x_fake), logits=x_fake))
    adversary_loss = 0.5*(adversary_loss_fake + adversary_loss_real)

    decoder_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(x_fake), logits=x_fake))

    all_variables = tf.trainable_variables()
    dc_var = [var for var in all_variables if 'dc_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]
    ad_var = [var for var in all_variables if 'ad_' in var.name]
    de_var = [var for var in all_variables if 'd_' in var.name]

    dn_var = [var for var in all_variables if 'dc_' in var.name or 'e_' in var.name]

    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_discriminator,
                                                     beta1=beta1).minimize(dc_loss, var_list=dc_var)

    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=en_var)
    adversary_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_discriminator,
                                                     beta1=beta1).minimize(adversary_loss, var_list=ad_var)                                                 
    decoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(decoder_loss, var_list=de_var)
    init = tf.global_variables_initializer()

    # Saving the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        if train_model:
            data = hyp_img['S']
            data = np.transpose(data, (1, 0))
            min_value = np.min(data)
            max_value = np.max(data)
            data = (data - min_value) / (max_value - min_value)
            saved_model_path, log_path, encoder_path, decoder_path= form_results()
            sess.run(init)
            for i in range(n_epochs+1):
                n_batches = (int)(num_examples / batch_size)
                print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                for b in range(n_batches):
                    z_real_dist = z_real['backnew']
                    min_value = np.min(z_real_dist)
                    max_value = np.max(z_real_dist)
                    z_real_dist = (z_real_dist - min_value) / (max_value - min_value)
                    batch_xr = data
                    batch_x = np.reshape(batch_xr[(b*batch_size):(b*batch_size+batch_size),:],[batch_size,bands])#作用是
                    sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    sess.run(discriminator_optimizer,
                                feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})
                    sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    sess.run(adversary_optimizer,
                                feed_dict={x_input: batch_x, x_target: batch_x})
                    sess.run(decoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x,})
                    e_output = sess.run(encoder_output, feed_dict={x_input: batch_x})#输出Z层的值
                    d_output = sess.run(decoder_output,feed_dict={x_input: batch_x})
                    if b % 1 == 0:
                        a_loss, d_loss, g_loss ,ad_loss , de_loss= sess.run(
                            [autoencoder_loss, dc_loss, generator_loss,adversary_loss,decoder_loss],
                            feed_dict={x_input: batch_x, x_target: batch_x,
                                    real_distribution: z_real_dist})

                    print("Epoch: {}, iteration: {}".format(i, b))
                    print("Encoder_Loss: {}".format(a_loss))
                    print("D1_Loss: {}".format(d_loss))
                    print("G2_Loss: {}".format(g_loss))
                    print("D2_Loss: {}".format(ad_loss))
                    print("Decoder_Loss: {}".format(de_loss))

                    if i % 100 == 0:
                        #d_output_x = tf.transpose(d_output,perm = [1,0])
                        savemat(encoder_path + 'x_encoder%d.mat'%(i), {'x_encoder': e_output})
                        savemat(decoder_path + 'x_decoder%d.mat'%(i), {'x_decoder': d_output})

                step += 1
                saver.save(sess, save_path=saved_model_path, global_step=1)
               
        else:
            data = hyp_img['S']
            data = np.transpose(data, (1, 0))
            min_value = np.min(data)
            max_value = np.max(data)
            data = (data - min_value) / (max_value - min_value)
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/Saved_models/'))
            z_real_dist = z_real['backnew']
            min_value = np.min(z_real_dist)
            max_value = np.max(z_real_dist)
            z_real_dist = (z_real_dist - min_value) / (max_value - min_value)
            value = data
            gen_target = []
            #批量生成,循环次数 = 需求数量/目标先验数量,AV数据集是1436/3=478
            for i in range(478):
                x_encoder = sess.run(encoder_output, feed_dict={x_input: value})
                x_decoder = sess.run(decoder_output, feed_dict={x_input: value})
                epsilon = tf.random_normal(shape=tf.shape(x_decoder), mean=0.0, stddev=0.01)
                x_decoder_with_noise = x_decoder + epsilon#加噪增加鲁棒性
                x_decoder = sess.run(x_decoder_with_noise, feed_dict={x_input: value})
                gen_target.append(x_decoder)
                value = x_decoder
            gen_target = tf.stack(gen_target, axis=0)
            gen_target = tf.reshape(gen_target, [-1, gen_target.shape[2]])
            print(gen_target.shape)
            a_loss, d_loss, g_loss ,ad_loss , de_loss= sess.run(
                [autoencoder_loss, dc_loss, generator_loss,adversary_loss,decoder_loss],
                feed_dict={x_input: data, x_target: data,
                        real_distribution: z_real_dist})
            #print(a,b)
            print("Encoder_Loss: {}".format(a_loss))
            print("D1_Loss: {}".format(d_loss))
            print("G2_Loss: {}".format(g_loss))
            print("D2_Loss: {}".format(ad_loss))
            print("Decoder_Loss: {}".format(de_loss))

            with tf.compat.v1.Session() as sess:
                # 将TensorFlow张量在会话中运行并获取结果
                gen_target = sess.run(gen_target)
                # 创建要保存的数据字典
                aa = {'gen_target_AV': gen_target}
                # 保存为MAT文件
                sio.savemat('gen_target_AV.mat', aa)

if __name__ == '__main__':
    #train(train_model=True)
    train(train_model=False)#False