#-*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class tensorflow(object):

    def inference(self,x:'input_data') -> 'model':
        '''学習モデル
        NNの学習モデル
        Args:
            x (list): input_data
        Returns:
            int: output model
        '''
        #入力層から中間層
        with tf.name_scope("hidden"):
            w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1), name="w1")
            b_1 = tf.Variable(tf.zeros([64]), name="b1")
            h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

            #中間層の重みの分布をログ出力
            tf.summary.histogram('w_1',w_1)

        #中間層から出力層
        with tf.name_scope("output"):
            w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), name="w2")
            b_2 = tf.Variable(tf.zeros([10]), name="b2")
            out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)
        
        return out

    def loss(self,out:'model',y:'actual_data')->'loss':
        '''loss funcstion
        calculate loss funstion
        Args:
            out(int):model output
        Returns:
            int: result of loss
        '''
        #誤差関数
        with tf.name_scope("loss"):    
            loss = tf.reduce_mean(tf.square(y - out))

            #誤差をログ出力
            tf.summary.scalar("loss", loss)

        return loss

    def train(self,loss:'loss')->'train_step':
        '''train
        training implementation
        Args:
            loss(int): value of loss
        Returns:
            int: result of training
        '''
        #訓練
        with tf.name_scope("train"):
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        return train_step

if __name__ == "__main__":
    t = tensorflow()

    #mnistデータを格納したオブジェクトを呼び出す
    mnist = input_data.read_data_sets("data/", one_hot=True)

    #入力データを定義
    x = tf.placeholder(tf.float32, [None, 784])

    #入力画像をログに出力(ミニバッチサイズ,縦,横,チャネル数)
    img = tf.reshape(x,[-1,28,28,1])
    tf.summary.image("input_data", img, 10)

    #正解データを定義
    y = tf.placeholder(tf.float32, [None, 10])

    out = t.inference(x)
    loss = t.loss(out,y)
    train_step = t.train(loss)

    #評価
    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        #精度をログ出力
        tf.summary.scalar("accuracy", accuracy)

    #変数の初期化
    init =tf.global_variables_initializer()
    #ログのマージオペレーション
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        #ログの出力先と対象のグラフを指定
        summary_writer = tf.summary.FileWriter("logs", sess.graph)
        sess.run(init)    

        #テストデータをロード    
        test_images = mnist.test.images    
        test_labels = mnist.test.labels    

        for step in range(1000):        
            train_images, train_labels = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

            if step % 10 == 0:
                #ログを取る処理を実行する（出力はログ情報が書かれたプロトコルバッファ）
                summary_str = sess.run(summary_op, feed_dict={x:test_images, y:test_labels})
                #ログ情報のプロトコルバッファを書き込む
                summary_writer.add_summary(summary_str, step)
