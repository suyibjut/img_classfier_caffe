# Created by wz on 17-5-12.
# encoding=utf-8
import sys

sys.path.append('/opt/caffe/python')  # 先将pycaffe 路径加入环境变量中
import caffe, cv2, numpy as np


class Classfier:  # 将模型封装入一个分类器类中
    def __init__(self, deploy, model, mu):
        self.net = caffe.Net(deploy, model, caffe.TEST)  # 初始化网络结构及其中的参数
        self.mu = mu

    def classify(self, img):
        img = (img - self.mu) * 0.00390625  # 减去均值后再进行缩放
        self.net.blobs['data'].data[...] = img  # 将图片数据送入data层的blobs
        out = self.net.forward()['prob']  # 执行前向计算，并得到最后prob层的输出结果
        return out


def main():
    mean_file = 'train.npy'
    mean = np.load(mean_file)  # 加载均值文件
    classifier = Classfier('deploy.prototxt', 'snapshot/alpha_iter_10000.caffemodel', mean)  # 创建我们的分类器
    with open('test.txt') as f:  # 读取测试集中的图片
        l = f.readlines()
    for i in l:
        print i
        name, label = i.split(' ')
        img = cv2.imread(name)  # 读取图片
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prob = classifier.classify(img)  # 使用分类器分类，得到概率
        print prob  # 输出概率值
        print chr(np.argmax(prob) + 65)  # 输出概率最大值对应的英文字母
        if np.argmax(prob) == int(label):
            cv2.imshow('img', img)  # 输出原始图片
            cv2.waitKey()  # 等待按键


if __name__ == '__main__':
    main()