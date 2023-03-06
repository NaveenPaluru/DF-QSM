from __future__ import print_function

import argparse
import csv
import os, logging
import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from model import Model
from padutils import*
import scipy.io 

parser = argparse.ArgumentParser(description='Susceptibility Correction')
parser.add_argument('--sgpu', default= 0, type=int, help='gpu index (start)')
parser.add_argument('--saveroot', default='./savedModels', type=str, help='save directory')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Model
print('==> Building model .. !')
net = Model()

# cuda
if use_cuda:
	torch.cuda.set_device(args.sgpu)
	net.cuda()
	print(torch.cuda.device_count())
	print('Using CUDA..')

# loading 
checkpoint = torch.load(os.path.join(args.saveroot, 'model.pth'))
net.load_state_dict(checkpoint)
net.eval()

# load train stats
data = scipy.io.loadmat('./data/tr-stats.mat')
if use_cuda:
    a  = torch.tensor(data['inp_mean']).cuda()
    b  = torch.tensor(data['out_mean']).cuda()    
    x  = torch.tensor(data['inp_std' ]).cuda()
    y  = torch.tensor(data['out_std' ]).cuda()      
else:
    a  = torch.tensor(data['inp_mean'])
    b  = torch.tensor(data['out_mean'])
    x  = torch.tensor(data['inp_std' ])
    y  = torch.tensor(data['out_std' ])    

# input

inp = scipy.io.loadmat('./data/phs.mat')
phs, N_dif, N_16 = padding_data(inp['phs'])
phs = (torch.tensor(phs).float())

out_data = './output/'

with torch.no_grad():	

	print('\nReconstructing Susceptibility from Local Field')

	if use_cuda:
		phs = phs.cuda()					
		phs = (phs-a)/x		                               
	else:			
		phs = (phs-a)/x
			
	outputs = net(phs)
	output = (outputs * y + b).detach().cpu().numpy()

	pred_sus = crop_data(output, N_dif)           
	mdic = {"susc" : pred_sus}
	filename = out_data + 'prediction.mat'
	scipy.io.savemat(filename, mdic)

print('\nDone...!')		 


