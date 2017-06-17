#!/usr/bin/python
#coding:utf-8

'the test moudle'

__author__ = 'Easo Chen'

from gen_captcha import gen_captcha_by_text
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import numpy as np
import tensorflow as tf

text, image = gen_captcha_by_text()
print("验证码图像channel:", image.shape)  # (60, 160, 3)

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)
print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐


# 把RGB图像转为灰度图像
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image【,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
# 所谓向量就是：例如向量[0,0,0,0,1,0,0,0,0,0]，就代表'4',因为第五个位置的值最大



char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN) # 长度为4*63=252的零向量 
    def char2pos(c):
        if c =='_':
            k = 62
            return k  #不在0-61范围内  k:   num:0-9,al:  10-35,AL : 36-61
        k = ord(c)-48 #ord函数:字符转ascii数值(十进制)
        if k > 9: #不是数字
            k = ord(c) - 55
            if k > 35: 
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
    for i, c in enumerate(text):  #enumerate函数用来遍历数组的下标和元素
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]  #vec.nonzero返回一个元组，表示各个非零元素的位置
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


#用来识别
def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
        i = 0
        for n in text:
                vector[i*CHAR_SET_LEN + n] = 1
                i += 1
        return vec2text(vector)

if __name__ == '__main__':

    text, image = gen_captcha_by_text()
    image = convert2gray(image)
    image = image.flatten() / 255
    predict_text = crack_captcha(image)
    print("正确: {}  预测: {}".format(text, predict_text))

