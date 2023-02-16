from utils.dataloaders import create_multilabel_classification_dataloader
import torch
import cca_core
import numpy as np
import argparse

def computer_cca(act1, act2):
    num_datapoints, channels, h, w = act1.shape
    temp1 = act1.reshape((num_datapoints * h * w, channels))
    num_datapoints, channels, h, w = act2.shape
    temp2 = act2.reshape((num_datapoints * h * w, channels))
    f_results = cca_core.get_cca_similarity(temp1.T, temp2.T, epsilon=1e-10, verbose=False)
    print('{:.4f}'.format(np.mean(f_results["cca_coef1"])))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default='./weights/tiny_rand_last.pt')
    parser.add_argument('--model2', type=str, default='./weights/tiny_rand_init')
    return parser.parse_known_args()[0] if known else parser.parse_args()

opt = parse_opt()
model1 = torch.load(opt.model1)
model1 = model1['model']
model2 = torch.load(opt.model2)
model2 = model2['model']

dataloader1 = create_multilabel_classification_dataloader(path='./data_plot/data',
                                                          csvpath='./data_plot/data.csv',
                                                          imgsz=224,
                                                          batch_size=16,
                                                          workers=8)
images, labels = next(iter(dataloader1))
images = images.cuda().half()

model1_layer = []
model2_layer = []

x = model2.conv1(images)
x = model2.m1(x)
model2_layer.append(x.cpu().numpy())
x = model2.conv2(x)
x = model2.m2(x)
model2_layer.append(x.cpu().numpy())
x = model2.conv3(x)
x = model2.m3(x)
model2_layer.append(x.cpu().numpy())
x = model2.conv4(x)
x = model2.m4(x)
model2_layer.append(x.cpu().numpy())

x = model1.conv1(images)
x = model1.m1(x)
model1_layer.append(x.cpu().numpy())
x = model1.conv2(x)
x = model1.m2(x)
model1_layer.append(x.cpu().numpy())
x = model1.conv3(x)
x = model1.m3(x)
model1_layer.append(x.cpu().numpy())
x = model1.conv4(x)
x = model1.m4(x)
model1_layer.append(x.cpu().numpy())

for i in range(len(model1_layer)):
    print(f"CCA similarity of layer/conv{i}:")
    computer_cca(model1_layer[i], model2_layer[i])

