#!/usr/bin/python
#coding:utf-8

'the captcha generator'

__author__ = 'Easo Chen'

from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import random,time

# 验证码字符
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


train_dir = '/home/easo/train'

#验证码长度为4个字符，宽度为160,高度为60（默认）
def get_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 生成字符对应的验证码
def gen_captcha_by_text():
    image = ImageCaptcha()

    captcha_text = get_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    image.write(captcha_text, train_dir+'/'+captcha_text + '.png')  # 生成png格式图片存到路径中

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    # 测试
    captcha_num = input("captcha nums:")
    for i in range(captcha_num):
        text, image = gen_captcha_by_text()
        # print 'begin ',time.ctime(),type(image)
        # f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
        # plt.imshow(image)


        #plt.show()
    print("end")

