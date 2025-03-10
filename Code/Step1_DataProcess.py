# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# 每个病灶向外围两个像素扩展， 并保存所有配准的框的最大范围， 叠成（层）*（长）*（宽）的矩阵
# 将原始的MRI图像转化成npy数据贮存， 获得含有文件路径和对应label的csv文件
# -

import os
from glob import glob
from tqdm import tqdm
import re
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import utils
from PIL import Image
import matplotlib.pyplot as plt
import h5py


# 注意：病灶（配准后数据）路径应为【 ../ 存放一级目录 / 医院名称 / 病灶类型 / 病灶ID 】
params = {
    'data_dir': '../Add_sample/Revise V3/',  # 数据集（已配准）路径 
    'all_path': '../Add_sample/',  # npy文件存放路径
    'save_text': '../Add_sample/Data_npy/label_add2.txt', # 包含所有文件以及对应的label
    'save_csv': '../Add_sample/Data_npy/label_add2.csv',  # 包含所有文件以及对应的label，用于dataloader
    'phases': ['T1', 'T2', 'AP', 'PP', '20 MIN'],
    'save_sample': False,  # 将样本名和对应期的路径保存
    'DEBUG': True,  # 显示相关文件信息
}


# + code_folding=[]
def load_df_file():
    # 记录标注文件
    data_dir = params['data_dir']# 数据集路径

    pat = os.path.join( data_dir, r'*/*/ROI/*.nrrd' )# glob寻找人工标注文件的模式
    label_filepath_list = glob(pat)# 所有的人工标注文件，列表
    print( '%d label nrrd file in total in %s.'%(len(label_filepath_list),data_dir) )

    sample_filepaths_df = pd.DataFrame(
        #index=label_filepath_list,
        columns=['label_fixed']+['label_'+phase for phase in params['phases']]+params['phases'])# 新建一个dataframe，用以保存配准信息
    sample_filepaths_df.index.name = 'sample_path'

    for label_filepath in label_filepath_list:
        label_path, label_filename = os.path.split( label_filepath )
        label_filename = label_filename.replace(' .nrrd', '.nrrd')
        label_filename = label_filename.replace('___.nrrd', '.nrrd')
        label_filename = label_filename.replace('__.nrrd', '.nrrd')
        label_filename = label_filename.replace('_.nrrd', '.nrrd')
        sample_id = os.path.dirname(label_path)
        label_phase = os.path.splitext(label_filename)[0]
        if 'M_' not in label_phase:
            # 配准时的基准序列
            sample_filepaths_df.loc[sample_id,'label_fixed'] = label_filepath
            sample_filepaths_df.loc[sample_id,'label_{}'.format(label_phase)] = label_filepath
        else:
            label_phase = label_phase.replace('M_','')
            sample_filepaths_df.loc[sample_id,'label_{}'.format(label_phase)] = label_filepath


    # 记录图像文件
    for sample_id in tqdm(sample_filepaths_df.index):
        # 图像文件路径
        image_path = os.path.join(sample_id,'Origin')
        # 找出所有moving图像文件路径：
        for phase in params['phases']:
            try:
                sample_filepaths_df.loc[sample_id,phase] = glob( os.path.join(image_path,'*{}*.nrrd'.format(phase)) )[0]
            except:
                None
                
        image_path = os.path.join(sample_id,'ORIGIN')
        # 找出所有moving图像文件路径：
        for phase in params['phases']:
            try:
                sample_filepaths_df.loc[sample_id,phase] = glob( os.path.join(image_path,'*{}*.nrrd'.format(phase)) )[0]
            except:
                None

    for sample_id in tqdm(sample_filepaths_df.index):
        for phase in params['phases']:
            try:
                if np.isnan(sample_filepaths_df.loc[sample_id, 'label_'+ phase]):
                    sample_filepaths_df.loc[sample_id, 'label_'+ phase] = sample_filepaths_df.loc[sample_id, 'label_fixed']
            except:
                None
                
    sample_filepaths_df.dropna(axis=0, inplace=True)
    return sample_filepaths_df

df_file = load_df_file()

# +
if params['DEBUG']:
    print(df_file.head())
    print(df_file.info())
    
# 保存文件
if params['save_sample']:
    df_file.to_csv('Sample_file_zsyy.csv')

# + run_control={"marked": true}
# main
for sample_id in tqdm(df_file.index):
    # 病灶路径应为【 ../ 存放一级目录 / 医院名称 / 病灶类型 / 病灶ID 】
    label_name = sample_id.split('/')[3]
    patient_id = sample_id.split('/')[4]
    hospital = sample_id.split('/')[2]

    # 对每一个期相：
    for phase in params['phases']:
        image_filepath = df_file.loc[sample_id, phase]
        image_sitk = sitk.ReadImage(image_filepath)
        image_array = sitk.GetArrayFromImage(image_sitk)

        # 如果一个mri图像包含有多个通道，应该把单通道提取出来
        if len(image_array.shape) > 3:
            dim = int(image_filepath.split('/')[-1][0]) - 1
            image_array = image_array[:, :, :, dim]

        label_filepath = df_file.loc[sample_id, 'label_' + phase]
        label_sitk = sitk.ReadImage(label_filepath)
        label_array = sitk.GetArrayFromImage(label_sitk)
        roi_labels = np.unique(label_array)[1:].tolist()  # 每个roi在掩膜中的像素值
        num_roi = len(roi_labels)  # roi数量

        roi_id = 0
        # 对每一个病灶
        for roi_label in roi_labels:
            roi_id += 1
            # 提取该roi的影像特征
            index = np.where(label_array == roi_label)
            out = image_array[min(index[0]):max(index[0]) + 1,
                              min(index[1]):max(index[1]) + 1,
                              min(index[2]):max(index[2]) + 1]
            np.save(
                '{save_path}{hospital}//{patient_id}_{label_name}_{roi_id}_{phase}.npy'
                .format(save_path=params['all_path'],
                        patient_id=patient_id,
                        hospital=hospital,
                        label_name=label_name,
                        roi_id=str(roi_id).zfill(2),
                        phase=phase), out)

            if roi_label == 1:
                label_01 = '1' if 'HCC' in image_filepath else '0'
                with open(params['save_text'], 'a') as f:
                    f.write(
                        '{save_path}{hospital}//{patient_id}_{label_name}_{roi_id}.npy\t{label01}\n'
                        .format(save_path=params['all_path'],
                                hospital=hospital,
                                patient_id=patient_id,
                                label_name=label_name,
                                roi_id=str(roi_id).zfill(2),
                                label01=label_01))


# + code_folding=[0]
def check_0dim():
    # 去掉含0的维数
    phase = ['_T1', '_T2', '_AP', '_PP', '_20 MIN']
    di = []
    for i in range(len(pd_txt)):
        name_file = pd_txt.path[i]

        try:

            for k in range(5):
            # 初始化
                    img_chan = np.load(name_file[:-4] + phase[k] + '.npy')
                    img_shape = img_chan.shape
                    a = img_shape[0]
                    c = img_shape[1]
                    d = img_shape[2]

                    if a*c*d == 0:
                        print(name_file)
                        di.append(i)
                        break
        except:
                 di.append(i)

    print('去除掉异常数据，该数据含0维数：', di)
    return di


# +
# 将txt文档转换为csv
pd_txt = pd.read_csv(params['save_text'], sep='\t')
pd_txt.columns = ['path', 'label']
pd_txt.drop_duplicates(inplace=True, ignore_index=True)

di = check_0dim()
pd_txt_drop = pd_txt.drop(pd_txt.index[di])
pd_txt_drop = pd_txt_drop.reset_index()
pd_txt_drop = pd_txt_drop.drop(columns=['index'])
pd_txt_drop.to_csv(params['save_csv'], index=False)

if params['DEBUG']:
    print(pd_txt_drop.head())
# -

# ALL_OUTSOURCE 整合所有外部数据到一个csv文件中
# 也可以整合多个版本的数据
try:
    pd_zs = pd.read_csv('../All_outsource_v5/ZSYY/label_ZSYY_v5.csv')
    pd_zj = pd.read_csv('../All_outsource_v5/ZJYY/label_ZJYY_v5.csv')
    pd_yd = pd.read_csv('../All_outsource_v5/YDYY/label_YDYY_v5.csv')
    pd_all = pd.concat([pd_yd, pd_zj, pd_zs], axis=0)
    pd_all.drop_duplicates(inplace=True, ignore_index=True)
    pd_all.reset_index(inplace=True)
    pd_all.drop(columns=['index'],inplace=True)
    pd_all.to_csv('../All_outsource_v5/label_Out_v5.csv')
    print(pd_all.head())
except:
    print('Please check the file exists!')
