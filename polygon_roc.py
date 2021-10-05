import similaritymeasures
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_curve,auc
from shapely.geometry import Polygon
import matplotlib.pyplot as plt



def test(list,header,np):
    pn = []
    score = []
    for i in list:
        num_data = pd.read_csv(header + str(i) + "_nes.csv").values
        per = Polygon(num_data)        
# print the results
        print(i)
        print(per.area)
        pn.append(np)
        fig = plt.figure()
        score.append( 1 - per.area/1.066)
#        score.append( 1 - per.area/2.0)
        x, y = per.exterior.xy
#        plt.fill(x, y, c="red")
        plt.xlim(0,1.1)
        plt.ylim(0,1.1)
        plt.plot(x, y, c="red",marker="o")
#        plt.plot(x[0], y[0], c="green",marker="D")
        #plt.scatter(num_data[0,:],num_data[1,:])
        fig.savefig("images2/chart_" + str(i) + ".png")
    return pn, score

#exp_data = pd.read_csv("normal_output/n3_nes.csv").values
normal_list = [1057,1058,1062,1063,1064,1065,1072,1073,1079,1080,1082,1084,1092,1093,1112,1114,1118,1119,1124,1126]
chd_list = [1,2,3,8,9,13,19,20,25,26,55,56,64,65,108,109,111,112,114,115]

print("normal")
#num = "_main"
num = "_main_result"
#num = "_sub_result4"
#num = "_simple5"
#num = "_simple_reg5"
nl_pn, nl_score = test(normal_list,"normal_output" + num + "/n",0)
print("chd")
cl_pn, cl_score = test(chd_list,"chd_output" + num + "/chd",1)
n_chd_list, score = normal_list + chd_list ,nl_score + cl_score 
fpr,tpr,a = roc_curve(nl_pn + cl_pn,nl_score + cl_score,pos_label=1)
for i in range(len(n_chd_list)):
    print(n_chd_list[i],score[i])
print(auc(fpr,tpr))
 
