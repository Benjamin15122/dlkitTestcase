import tensorflow as tf
from mnist import IMAGE_PIXELS
import math

HIDDEN_UNITS = 100
LEARNING_RATE = 0.01

def predict(data):
    # Define the model
     # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS, HIDDEN_UNITS],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([HIDDEN_UNITS]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [HIDDEN_UNITS, 10],
            stddev=1.0 / math.sqrt(100)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    opt = tf.train.AdamOptimizer(LEARNING_RATE)

    keep_prob = tf.placeholder(tf.float32)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "/workspace/model/mnist.ckpt")
        print ("Model restored.")
       
        prediction=tf.argmax(y,1)
        return prediction.eval(feed_dict={x: [data],keep_prob: 1.0}, session=sess)



def predict_handler(event, context):
    data = event['data']['data']
    pred = predict(data)
    print(pred)
    return str(pred[0])


if __name__ == '__main__':
    main()