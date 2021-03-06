import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
# 验证码基本信息
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOW_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
           'V', 'W', 'X', 'Y', 'Z']
CAPTCHA_LIST = NUMBER + LOW_CASE + UP_CASE
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160

# 随机生成验证码文本
def random_captcha_text(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LEN):
    '''
    :param char_set:
    :param captcha_size:
    :return:
    '''
    captcha_text = [random.choice(char_set) for _ in range(captcha_size)]
    return ''.join(captcha_text)

# 生成随机验证码
def gen_captcha_text_and_image(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT, save=None):
    '''
    :param width:
    :param height:
    :param save:
    :return: np数组
    '''
    image = ImageCaptcha(width=width, height=height)
    # 验证码文本
    captcha_text = random_captcha_text()
    captcha = image.generate(captcha_text)
    # 保存
    if save: image.write(captcha_text, captcha_text + '.jpg')
    captcha_image = Image.open(captcha)
    # 转化为np数组
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

if __name__ == '__main__':
    a = gen_captcha_text_and_image(CAPTCHA_WIDTH, CAPTCHA_HEIGHT, save=False)
    print(a[0])
    plt.imshow(a[1])
    plt.show()