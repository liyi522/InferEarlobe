# -*- coding: utf-8 -*-
import argparse
import cv2
import os
import torch
import numpy as np
import pandas as pd
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#--------------------------
# part1: 耳垂区域图像截取
# 进行检测
detect_dir = 'data/images'
outdir = 'runs'
pb1 = 'models/best.pb'
cmd1 = 'python detect.py --source data/images --weights models/best.pb --max-det 1 --save-txt --save-conf --save-crop --device cpu'
os.system(cmd1)
crop_out = 'runs/crops/earlobe'
lab_out = 'runs/labels'
# 文件载入、耳垂目标检测以及耳垂截取
imgfils = os.listdir(detect_dir)
# 检查cmd1命令是否运行完了
fil_num = len(os.listdir('runs'))
while fil_num < (len(imgfils)+2):
    time.sleep(10)
# 将原始imgfils里面的文件后缀都修改成.jpg
for i in range(len(imgfils)):
    tmp = os.path.splitext(imgfils[i])[0] + '.jpg'
    imgfils[i] = tmp

rlt1 =  pd.DataFrame()
lobe_num = 0
for i in range(len(imgfils)):
    img_name = os.path.splitext(imgfils[i])[0]
    if os.path.exists(crop_out + '/' + imgfils[i]) == False:
        tmp_da = pd.DataFrame(np.zeros((1, 7)), columns = ['name','xc_std', 'yc_std', 'w_std', 'h_std', 'confidence', 'class'])
    else:
        with open(lab_out + '/' + img_name + '.txt', 'r') as f:
            read_da = f.readlines()[0].strip('\n').split(' ')
        read_da = read_da[1:]
        read_da.insert(0,imgfils[i])
        read_da.append('earlobe')
        tmp_da = pd.DataFrame(np.array((read_da)).reshape(1,7), columns = ['name','xc_std', 'yc_std', 'w_std', 'h_std', 'confidence', 'class'])
        lobe_num += 1
    rlt1 = rlt1.append(tmp_da)

rlt1['name'] = imgfils
rlt1['class'] = ['earlobe' for x in range(len(imgfils))]
rlt1.to_csv(os.path.join(outdir,'CropEarLobe_info.csv'))

#--------------------------
IMAGE_SIZE = 224
#--------------------------
# part2: 对耳垂区域图像进行耳垂表型分类
# 耳垂图像输入pb模型，输出分类结果
def flowers_test(pb, nclass, imgs):
        # 两 runs 之间清一次计算图
        tf.reset_default_graph()
        assert os.path.exists(pb)
        with tf.gfile.FastGFile(pb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def,name='')
        with tf.Session() as sess:
            #print(tf.get_default_graph().get_operations())
            input_x = sess.graph.get_tensor_by_name('input:0')
            out = sess.graph.get_tensor_by_name('pre:0')
            ret = sess.run(out,feed_dict={input_x:imgs}) # 输入是(?x224x224x3)，因此批量输入是合理的
        rlt = []
        for i in range(1,len(ret)//3+1):
            tmp_sum = ret[i*3-3]+ret[i*3-2]+ret[i*3-1] # sum the probability of >= k class
            rlt.append(round(tmp_sum))    # round result
        return rlt


#循环的话仍旧使用上面的imgs循环，这样可以先判断在earlobe文件中存在不，不存在直接输出一个0数据行，否则输出模型的分类结果。
# image data load
imgs = np.empty((lobe_num, 224, 224, 3))
count = 0
earlobe = []
pres = np.zeros((1,len(imgfils)))
for i in range(len(imgfils)):
    lobe_dir = os.path.join(crop_out, imgfils[i])
    if os.path.exists(lobe_dir):
        imge= cv2.resize(cv2.imread(lobe_dir), (224, 224))
        imgs[count:,:,:] = imge
        count += 1
        earlobe.append(1)
    else:
        earlobe.append(0)

# 
stds =[['free', 'partially attached', 'attached'], ['small', 'medium', 'large'], ['circular', 'square', 'triangular']]
phens = ['lobe attachment', 'lobe size', 'lobe shape']
pbs = ['LALacc100','LSLacc100','LTLacc100']
# stds = [['free', 'partially-attached', 'attached']]
# phens = ['lobe-attachment']
# pbs = ['LAL50']
num_classes = [3,3,3]
rlt_name = ['Imgid']
rlt_name.extend(phe for phe in phens)
rlt_name.extend([phe+'-cls' for phe in phens])
pb_rlt = pd.DataFrame(np.zeros((len(imgfils), len(pbs)*2+1)), columns = rlt_name)
pb_rlt['Imgid'] = imgfils
pddir = 'models'
for n in range(len(pbs)):
    pb_use = os.path.join(pddir, pbs[n] + '.pb')
    rlt2 = flowers_test(pb_use, num_classes[n], imgs)
    rlt3 = []
    rlt4 = []
    count = 0
    for m in earlobe:
        if m == 0:
            rlt3.append('NA')
            rlt4.append(-9)
        else:
            rlt3.append(stds[n][rlt2[count]-1])
            rlt4.append(rlt2[count])
            count += 1
    pb_rlt[phens[n]] = rlt3
    pb_rlt[phens[n]+'-cls'] = rlt4

pb_rlt.to_csv(os.path.join(outdir,'Predicted_phenos.csv'))
#--------------------------
if __name__ == '__main__':
    time.sleep(10)
    print('------------------------------')