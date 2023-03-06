#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:01:32 2020

@author: cds


"""


import torch
#from QSMnet import QSMnet
import numpy as np


def dipole_kernel(matrix_size, voxel_size, B0_dir=[0,0,1]):
    [Y,X,Z] = np.meshgrid(np.linspace(-np.int(matrix_size[1]/2),np.int(matrix_size[1]/2)-1, matrix_size[1]),
                       np.linspace(-np.int(matrix_size[0]/2),np.int(matrix_size[0]/2)-1, matrix_size[0]),
                       np.linspace(-np.int(matrix_size[2]/2),np.int(matrix_size[2]/2)-1, matrix_size[2]))
    X = X/(matrix_size[0])*voxel_size[0]
    Y = Y/(matrix_size[1])*voxel_size[1]
    Z = Z/(matrix_size[2])*voxel_size[2]
    D = 1/3 - np.divide(np.square(X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2]), np.square(X)+np.square(Y)+np.square(Z) + np.finfo(float).eps )
    D = np.where(np.isnan(D),0,D)

    D = np.roll(D,np.int(np.floor(matrix_size[0]/2)),axis=0)
    D = np.roll(D,np.int(np.floor(matrix_size[1]/2)),axis=1)
    D = np.roll(D,np.int(np.floor(matrix_size[2]/2)),axis=2)
    D = torch.tensor(D)    
    return D
    
def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N,16.))*16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0],N_dif[0]),(N_dif[1],N_dif[1]),(N_dif[2],N_dif[2]))
    pad_field = np.pad(input_field, pad_width = npad, mode = 'constant', constant_values = 0)
    pad_field = np.expand_dims(pad_field, axis=0)
    pad_field = np.expand_dims(pad_field, axis=0)
    return pad_field, N_dif, N_16

def crop_data(result_pad, N_dif):
    result_pad = result_pad.squeeze()
    N_p = np.shape(result_pad)
    result_final  = result_pad[N_dif[0]:N_p[0]-N_dif[0],N_dif[1]:N_p[1]-N_dif[1],N_dif[2]:N_p[2]-N_dif[2]]
    return result_final


if __name__=="__main__":
    net = QSMnet().cuda(0)
    inp = np.random.rand(170,170,160)
    print(inp.shape)
    inp, N_dif, N_16 = padding_data(inp)
    print(inp.shape)
    inp = torch.tensor(inp).float()
    print(inp.shape)
    inp = inp.cuda(0)
    out = net(inp).squeeze().detach().cpu().numpy()
    print(out.shape)
    out = crop_data(out, N_dif)
    print(out.shape)