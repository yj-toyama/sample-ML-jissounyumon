import tensorflow as tf

dataset = tf.contrib.data.Dataset.range(10)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
#インタラクティブセッションとしてセッション定義
sess = tf.InteractiveSession()
for i in range(10):
    print(sess.run(next_element))