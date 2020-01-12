#-*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class tensorflow(object):

    def __init__(self):
        '''初期化関数
        必要データの初期化
        Args:
            None
        Return:
            None
        '''        
        #mnistデータを格納したオブジェクトを呼び出す
        mnist = input_data.read_data_sets("data/", one_hot=True)

    def inference(self,input:'inputdata')->'model output':
        '''モデル学習
        モデルの学習を行う
        Args:
            input(int): inputdata
        Return:
            int: model output
        '''
        #ユニット数(中間層)128個のLSTMセル
        #三段に積む
        stacked_cells = []
        for i in range(2):
            stacked_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=128))
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cells)

        outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32)

        #3階テンソルを2階テンソルのリストに変換
        #outputsは[バッチサイズ、時間長、出力長]となっているので、最終時間の値をスライスする
        outputs_list = tf.unstack(outputs, axis=1)
        #最終時系列情報を取得
        last_output = outputs_list[-1]

        w = tf.Variable(tf.truncated_normal([128,10], stddev=0.1))
        b = tf.Variable(tf.zeros([10]))

        out = tf.nn.softmax(tf.matmul(last_output, w ) + b)
        return out

    def loss(self,out:'model output',y:'actual data')->'loss':
        '''loss function
        calculate loss function
        Args:
            out(int): model output
            y(int): actual values
        Return:
            int: value of loss
        '''
        #正解データの型を定義
        y = tf.placeholder(tf.float32, [None, 10])
        #誤差関数（クロスエントロピー）
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), axis=[1]))
        return loss

    def train(self,loss:'loss')->'train_step':
        '''train
        train and learn
        Args:
            loss(int): loss
        Return:
            int: train_step
        '''
        #訓練
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        return train_step

if __name__ == "__main__":
    t = tensorflow()
    #入力データ整形
    num_seq = 28
    num_input = 28
    mnist = t.mnist

    x = tf.placeholder(tf.float32, [None, 784])
    #(バッチサイズ,高さ, 幅)の2階テンソルに変換
    input = tf.reshape(x, [-1, num_seq, num_input])
    #正解データの型を定義
    y = tf.placeholder(tf.float32, [None, 10])

    out = t.inference(input)
    loss = t.loss(out,y)
    train_step = t.train(loss)

    #評価
    correct = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        #テストデータをロード
        test_images = mnist.test.images
        test_labels = mnist.test.labels

        for i in range(1000):
            step = i+1
            train_images, train_labels = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x:train_images ,y:train_labels})

            #10階ごとに精度を検証
            if step % 100 == 0:
                acc_val = sess.run( accuracy, feed_dict={x:test_images, y:test_labels})
                print('Step %d: accuracy = %.2f' % (step, acc_val))