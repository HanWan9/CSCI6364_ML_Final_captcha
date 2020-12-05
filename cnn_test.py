# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from cnn_train import cnn_graph
from captcha_create import gen_captcha_text_and_image
from captcha_process import vec2text, convert2gray
from captcha_process import CAPTCHA_LIST, CAPTCHA_WIDTH, CAPTCHA_HEIGHT, CAPTCHA_LEN
import matplotlib.pyplot as plt

# 验证码图片转化为文本
def captcha2text(image_list, height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH):
    '''
    :param image_list:
    :param height:
    :param width:
    :return:
    '''
    x = tf.placeholder(tf.float32, [None, height * width])
    keep_prob = tf.placeholder(tf.float32)
    y_conv = cnn_graph(x, keep_prob, (height, width))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(y_conv, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), 2)
        text_list = []
        for image in image_list:
            vector_list = sess.run(predict, feed_dict={x: image, keep_prob: 1})
            vector_list = vector_list.tolist()
            text = [vec2text(vector) for vector in vector_list]
            text_list.append(text)

        return text_list[0]

if __name__ == '__main__':
    # text, image = gen_captcha_text_and_image()
    # image = convert2gray(image)
    # image = image.flatten() / 255
    # pre_text = captcha2text([image])
    # print('Label:', text, ' Predict:', pre_text)
    accurance = [0]
    text_list = []
    image_list = []

    # change the number of test samples:
    samples = 1000

    for _ in range(samples):
        text, image = gen_captcha_text_and_image()
        image = convert2gray(image)
        image = image.flatten() / 255
        text_list.append(text)
        image_list.append(image)

    pre_text_list = captcha2text([image_list])
    it = 1
    sample = []
    for x,y in zip(text_list,pre_text_list):
        sample.append(it)
        it += 1
        if x == y:
            accurance.append(accurance[-1]+1)
        else:
            accurance.append(accurance[-1]+0)
    accurance = accurance[1:]
    accurance = [accurance[_]/(_+1) for _ in range(len(accurance)) ]
    print(samples,"test samples: accurance:", accurance[-1])
    plt.plot(sample, accurance, color='blue', linewidth=1.0)
    plt.title('Accurance')
    plt.xlabel('Samples')
    plt.ylabel('Accurance')
    plt.show()


