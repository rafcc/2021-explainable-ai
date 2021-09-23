import csv
import os
import numpy as np
from itertools import chain
class seqiter:
    def pattern(self,data_array):
        pattern = []
        for i in self.pattern_index:
            if(self.binary == True):
                if(float(data_array[i]) > 0 ):
                    data_array[i] = 1
                else:
                    data_array[i] = 0
            pattern.append(float(data_array[i]))
        return pattern
    def flattenf(self,data):
        result = []
        for sublist in data:
            for item in sublist:
                result.append(item)
        return result
    def flattenc(self,data):
        result = []
        for sublist in data:
            for item in sublist:
                for co in item:
                    result.append(co)
        return result
    def pattern_cord(self,data_array,flag=0):
        pattern = []
        for i in self.pattern_index:
            if(flag == 0):
                if(data_array[i] == '0'):
                    tmp_l = [0.0,0.0,0.0,0.0]
                else:
                    tmp_l = data_array[i].split(':')
                    tmp_l = [float(s) for s in tmp_l]
                    if(self.size != None):
                        for i in range(4):
                            tmp_l[i] = tmp_l[i]*self.size[i]
                pattern.append(tmp_l)
        return pattern            
    def angle_solver(row):
        return None # under development
    def __init__(self,p_dirpath,c_dirpath,data_size = [0,2000],pattern = 1,angle_solv = False,reset = True,binary = True,size = None,filepath = None,flatten = True,stride_size = 1,kernel_size = 4):
        self.p_dirpath = p_dirpath
        self.c_dirpath = c_dirpath
        self.n_max = 0
        self.n_index = 0
        self.s_index = 0
        self.binary = binary
        self.reset = reset
        self.stride_size = stride_size
        self.kernel_size = kernel_size
        self.flatten = flatten
        if(size != None):
            self.size = size
        self.angle_pattern = [(1,18),(4,19),(5,20)]
        if(pattern == 1):
            self.pattern_index = [0,1,2,4,5,7,8,9,10]
        if(pattern == 2):
            self.pattern_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        if(pattern == 3):
            self.pattern_index = [0,1,2,4,5,7,8,9,10,11,15,17]
        if(pattern == 4):
            self.pattern_index = [0,1]            
        self.data = []
        for i in range(data_size[0],data_size[1]):
            p_path = self.p_dirpath + "/n" + str(i) + ".csv"
            c_path = self.c_dirpath + "/n" + str(i) + ".csv"
            print("loading:n" + str(i))
            if(os.path.exists(p_path) and os.path.exists(c_path)):
                self.n_max = self.n_max + 1
                self.data.append([[],[],[]])
                self.data[self.n_max - 1][0] = int(i)#num
                with open(p_path,'r') as f:
                    tmp = []
                    reader = csv.reader(f)
                    for row in reader:
                        if(angle_solv == True):
                            row = self.angle_solver(row)
                        tmp.append(self.pattern(row))
                    self.data[self.n_max - 1][1] = tmp#p
                with open(c_path,'r') as f:
                    tmp = []
                    reader = csv.reader(f)
                    for row in reader:
                        if(angle_solv == True):
                            row = self.angle_solver(row)
                        tmp.append(self.pattern_cord(row))
                    self.data[self.n_max - 1][2] = tmp#c

    def single_pick(self,data,s):
        pick_p = []
        pick_c = []
        for k in range(self.kernel_size):
            try:
                pick_p.append(data[1][s + k])
                pick_c.append(data[2][s + k])
            except:
                pick_p = None
                pick_c = None
        if(pick_p != None and pick_c != None and self.flatten == True):
            result = self.flattenf(pick_p) + self.flattenc(pick_c)
#            print(self.flattenf(pick_p),self.flattenc(pick_c))
        else:
            result = None
        return result
    def __iter__(self):
        return self
    def next(self):
        seg = []
        tmp = self.single_pick(self.data[self.n_index],self.s_index)
        if(tmp != None):
            self.s_index = self.s_index + self.stride_size
            return tmp
        if(tmp == None):
            self.n_index = self.n_index + 1
            self.s_index = 0
            try:
                tmp =  self.single_pick(self.data[self.n_index],self.s_index) 
                return tmp
            except:
                tmp = None
            if(tmp == None ):
                if(self.reset == True):
                    self.n_index = 0
                    tmp = self.single_pick(self.data[self.n_index],self.s_index)
                    return tmp
                else:
                    raise StopIteration
