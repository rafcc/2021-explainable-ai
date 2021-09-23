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
    def __init__(self,dirpath,pattern = 1,max_files=200,dir_or_file="dir",reset = True,binary = True,filepath = None,flatten = True,stride_size = 1,kernel_size = 4):
        self.dirpath = dirpath
        self.n_max = 0
        self.n_index = 0
        self.s_index = 0
        self.binary = binary
        self.reset = reset
        self.stride_size = stride_size
        self.kernel_size = kernel_size
        self.flatten = flatten
        if(pattern == 1):
            self.pattern_index = [0,1,2,4,5,7,8,9,10]
        if(pattern == 2):
            self.pattern_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        self.data = []
        if(dir_or_file == "dir"):
            for i in range(max_files):
                path = self.dirpath + "/n" + str(i) + ".csv"
                if(os.path.exists(path)):
                    self.n_max = self.n_max + 1
                    print(i)
                    with open(path,'r') as f:
                        tmp = []
                        reader = csv.reader(f)
                        for row in reader:
                            tmp.append(self.pattern(row))
                        self.data.append(tmp)
        if(dir_or_file == "file"):
                path = self.dirpath
                if(os.path.exists(path)):
                    with open(path,'r') as f:
                        tmp = []
                        reader = csv.reader(f)
                        for row in reader:
                            tmp.append(self.pattern(row))
                        self.data.append(tmp)

    def single_pick(self,data_array,s):
        pick = []
        for k in range(self.kernel_size):
            try:
                pick.append(data_array[s + k])
            except:
                pick = None
        if(pick != None and self.flatten == True):
            pick = self.flattenf(pick)
        return pick
    def __iter__(self):
        return self
    def next(self):
        seg = []
        tmp = self.single_pick(self.data[self.n_index],self.s_index)
#        print(self.s_index,self.n_index)
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
