import os
import random
import shutil

list_img = []
data_size = 0
data_size_dogs = 0
data_size_cats = 0
list_train_cats = []
list_train_dogs = []
list_val_cats = []
list_val_dogs = []

for file in os.listdir('./dataset/sub'): # 遍历文件夹
    list_img.append(file) # 将图片路径和文件名添加至image list
    data_size += 1

random.shuffle(list_img)#打乱数据集的顺序

for img in list_img:
    name = img.split(sep='.')#分割文件名
    image_type = name[0]
    if image_type == 'cat':
        data_size_cats += 1
        if data_size_cats <= 2500:
            list_val_cats.append(img)
        else:
            list_train_cats.append(img)
    else:
        data_size_dogs += 1
        if data_size_dogs <= 2500:
            list_val_dogs.append(img)
        else:
            list_train_dogs.append(img)

for img_cats_train in list_train_cats:
    shutil.copyfile('./dataset/sub/' + img_cats_train, './dataset/train/cats/' + img_cats_train)
for img_cats_val in list_val_cats:
    shutil.copyfile('./dataset/sub/' + img_cats_val, './dataset/val/cats/' + img_cats_val)
for img_dogs_train in list_train_dogs:
    shutil.copyfile('./dataset/sub/' + img_dogs_train, './dataset/train/dogs/' + img_dogs_train)
for img_dogs_val in list_val_dogs:
    shutil.copyfile('./dataset/sub/' + img_dogs_val, './dataset/val/dogs/' + img_dogs_val)