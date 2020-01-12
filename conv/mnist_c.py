#-*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class tensormodel(object):
    def __init__(self:'classobject') -> 'None':
        '''初期化関数
        Args:
            None
        Returns:
            None
        '''
        pass

    def inference(self:'classobject',img:'feature_data') -> 'model':
        '''
        学習モデル
        Args:
            None
        Returns:
            int: result of return model
        '''

        #畳み込み層1
        #フィルター定義(shape:縦,横,入力チャネル,出力チャネル)
        f1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
        #フィルター適用(shape:バッチ方向幅,縦,横,チャネル方向幅)
        conv1 = tf.nn.conv2d(img, f1, strides=[1,1,1,1], padding='SAME')
        b1 = tf.Variable(tf.constant(0.1, shape=[32]))
        h_conv1 = tf.nn.relu(conv1+b1)
        #プーリング層1:ブーリング層でストライドを2としているので、縦横サイズが1/2になる
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        #畳み込み層2
        f2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
        conv2 = tf.nn.conv2d(h_pool1, f2, strides=[1,1,1,1], padding='SAME')
        b2 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_conv2 = tf.nn.relu(conv2+b2)
        #プーリング層2:ブーリング層でストライドを2としているので、縦横サイズが1/2になる
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        #畳み込まれているものをフラットな形に変換
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        #全結合層
        w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        #出力層
        w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        out = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)
        return out


    def loss(self,out:'model',y:'actual data') -> 'loss':
        '''誤差関数
        Args:
            out (int): result of model
        Returns:
            int: result of calculate lossfuncton
        '''
        #誤差関数（クロスエントロピー）
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out + 1e-5), axis=[1]))
        return loss

    def train(self,loss:'loss') -> 'trainstep':
        '''
        学習
        Args:
            loss (int): reuslt of calcurate lossfunsction
        Returns:
            int: result of training
        '''
        #訓練
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        return train_step



if __name__ == "__main__":
    t = tensormodel()

    #mnistデータを格納したオブジェクトを呼び出す
    mnist = input_data.read_data_sets("data/", one_hot=True)

    #入力データを定義(shape:ミニバッチサイズ,28*28)
    x = tf.placeholder(tf.float32, [None, 784])
    #入力データを画像郡として扱うために整形(shape:ミニバッチサイズ,縦,横,チャネル数)
    img = tf.reshape(x,[-1,28,28,1])
    #正解データの型を定義
    y = tf.placeholder(tf.float32, [None, 10])

    out = t.inference(img)
    loss = t.loss(out,y)
    train_step = t.train(loss)

    #評価オペレータ
    correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    #変数の初期化オペレータ
    init =tf.global_variables_initializer()

    """実行部分"""
    with tf.Session() as sess:
        sess.run(init)
        #テストデータをロード
        test_images = mnist.test.images
        test_labels = mnist.test.labels

        #0-999までの1000ステップ実行
        for step in range(1000):
            train_images, train_labels = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

            #10回ごとに精度を検証
            if step % 100 == 0:
                acc_val = sess.run( accuracy, feed_dict={x:test_images, y:test_labels})
                print('Step %d: accuracy = %.2f' % (step, acc_val))