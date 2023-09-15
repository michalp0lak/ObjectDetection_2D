import glob, os
import random
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
import json
import shutil
import numpy as np


def India_LP_transformer(dataset_path, data_splits = {'training': 0.7, 'validation': 0.2, 'testing': 0.1}):

    files = [f for f in glob.glob(dataset_path + "/images/*.png")]

    random.shuffle(files)

    train, test = train_test_split(files,  test_size = round(1 - data_splits['training'],5), random_state=1)

    val, test = train_test_split(test, test_size = data_splits['testing']
                            /(data_splits['testing'] + data_splits['validation']))

    datasets = {'training': train, 'validation': val, 'testing': test}

    for dts in ['training', 'validation', 'testing']:
        if not os.path.exists(dataset_path + dts + '/'):
            os.makedirs(dataset_path + dts + '/')
            
        for item in datasets[dts]:

            itemname = item.split('/')[-1].split('.')[0]
            anotname = dataset_path + '/annotations/' + itemname + '.xml'

            if os.path.exists(anotname):

                with open(anotname, 'r') as f:
                    data = f.read()
           
                Bs_data = BeautifulSoup(data, "xml")
                xmin = int(re.findall(r'\d+',str(Bs_data.find('xmin')))[0])
                ymin = int(re.findall(r'\d+',str(Bs_data.find('ymin')))[0])
                xmax = int(re.findall(r'\d+',str(Bs_data.find('xmax')))[0])
                ymax = int(re.findall(r'\d+',str(Bs_data.find('ymax')))[0])

                annot = [{'class': 1, 'box': [xmin, ymin, xmax, ymax]}]

                with open(dataset_path + dts + '/' + itemname  + ".json", "w") as outfile:
                    json.dump(annot, outfile) 
                
                shutil.copyfile(item, dataset_path + dts + '/' + itemname  + ".png")

def Netherlands_LP_transformer(dataset_path, data_splits = {'training': 0.7, 'validation': 0.2, 'testing': 0.1}):

    files = [f for f in glob.glob(dataset_path + "*.jpeg")]

    random.shuffle(files)

    train, test = train_test_split(files,  test_size = round(1 - data_splits['training'],5), random_state=1)

    val, test = train_test_split(test, test_size = data_splits['testing']
                            /(data_splits['testing'] + data_splits['validation']))

    datasets = {'training': train, 'validation': val, 'testing': test}

    for dts in ['training', 'validation', 'testing']:

        if not os.path.exists(dataset_path + dts + '/'):
            os.makedirs(dataset_path + dts + '/')
            
        for item in datasets[dts]:

            itemname = item.split('/')[-1].split('.')[0]
            anotname = dataset_path + itemname + '.xml'

            if os.path.exists(anotname):

                with open(anotname, 'r') as f:
                    data = f.read()

                Bs_data = BeautifulSoup(data, "xml")
                xmin = int(re.findall(r'\d+',str(Bs_data.find('xmin')))[0])
                ymin = int(re.findall(r'\d+',str(Bs_data.find('ymin')))[0])
                xmax = int(re.findall(r'\d+',str(Bs_data.find('xmax')))[0])
                ymax = int(re.findall(r'\d+',str(Bs_data.find('ymax')))[0])

                annot = [{'class': 1, 'box': [xmin, ymin, xmax, ymax]}]

                with open(dataset_path + dts + '/' + itemname  + ".json", "w") as outfile:
                    json.dump(annot, outfile) 
                
                shutil.copyfile(item, dataset_path + dts + '/' + itemname  + ".png")


def USAEU_LP_transformer(dataset_path):

    splits = ['train', 'valid', 'test']
    datasets = ['training', 'validation', 'testing']

    for i, split in enumerate(splits):

        if not os.path.exists(dataset_path + datasets[i] + '/'):
            os.makedirs(dataset_path+ datasets[i] + '/')

        split_path = dataset_path + split + '/'
        anot_file = split_path + '_annotations.coco.json'

        with open(anot_file,"r") as f: annotations = json.loads(f.read())

        for image in annotations["images"]:

            anot_list = [anot for anot in annotations['annotations'] if (anot['image_id'] == image['id']) and (anot['category_id'] == 1)]
            anot_data = []
            itemname = image['file_name'].split('.jpg')[0]

            if len(anot_list) > 0:

                for anot_item in anot_list:
                            
                    box = np.array([anot_item['bbox'][0], anot_item['bbox'][1],
                                    anot_item['bbox'][0]+anot_item['bbox'][2], 
                                    anot_item['bbox'][1]+anot_item['bbox'][3]]).astype(np.int32).tolist()

                    anot_data.append({'class': anot_item['category_id'], 'box': box})

                with open(dataset_path + datasets[i] + '/' + itemname  + ".json", "w") as outfile:
                    json.dump(anot_data, outfile) 

                shutil.copyfile(split_path+ image['file_name'], dataset_path + datasets[i] + '/' + itemname  + ".png")