import numpy as np
import math
from processdata import Encode
from numpy import *
import time

class APTrain:
    def __init__(self):
        self.weight_matrix = np.array((0,0))
        self.sum_weight_matrix = np.array((0,0))
        self.last_update_weight = []
        self.cor = 0
        self.total = 0
        self.loss = 0
        self.step = 0

    def create_weight_matrix(self,depth,width):
        self.weight_matrix = np.zeros((depth,width))
        self.sum_weight_matrix = np.zeros((depth,width))
        self.last_update_weight = [0 for i in range(depth)]

    def train(self,parameter,train_encodes,dev_encodes,test_encodes):
        self.step = 1
        for i in range(parameter.ap_iter_num):
            print('第%d轮迭代：'%(i+1))
            starttime = time.time()
            self.total = self.cor = self.loss = left_bound = 0
            right_bound = parameter.ap_batch_size
            max_len = len(train_encodes)
            while left_bound<max_len:    #batch
                outputs,gold_labels = self.forward(train_encodes[left_bound:right_bound],parameter.class_num)
                self.backward(outputs,gold_labels,train_encodes[left_bound:right_bound])
                left_bound += parameter.ap_batch_size
                right_bound += parameter.ap_batch_size
                if right_bound >= max_len:
                    right_bound = max_len - 1
            for i in range(parameter.depth):
                times = self.step - self.last_update_weight[i]
                self.sum_weight_matrix[i] += self.weight_matrix[i] * times
                self.last_update_weight[i] = self.step
            print('训练时间：',time.time()-starttime)
            print('train accuarcy:',self.cor/self.total)
            print('loss:',self.loss )
            self.eval(dev_encodes, 'dev',parameter)
            # self.eval(test_encodes, 'test')
            if self.cor/self.total == 1.0:
                break
            train_encodes = self.encode_random(train_encodes)
        print('-------------------------')

    def forward(self,encodes,class_num):
        result_labels = []
        gold_labels = []
        for encode in encodes:
            sum = np.array([0.0 for i in range(class_num)])
            for e in encode.code_list:
                sum += self.weight_matrix[e]
            result_labels.append(sum)
            gold_labels.append(encode.label)
        return result_labels,gold_labels

    def backward(self,outputs,gold_labels,encodes):
        for i in range(len(outputs)):
            if self.get_maxIndex(outputs[i]) != self.get_maxIndex(gold_labels[i]):
                for wi in encodes[i].code_list:
                    times = self.step - self.last_update_weight[wi]
                    self.sum_weight_matrix[wi] += self.weight_matrix[wi]*times
                    self.weight_matrix[wi][self.get_maxIndex(outputs[i])] -= 1
                    self.weight_matrix[wi][self.get_maxIndex(gold_labels[i])] += 1
                    self.sum_weight_matrix[wi] += self.weight_matrix[wi]
                for wi in encodes[i].code_list:
                    self.last_update_weight[wi] = self.step
                self.loss += 1
            else:
                self.cor += 1
            self.total += 1
            self.step += 1

    def get_max(self,list):
        max = list[0]
        for i in range(len(list)):
            if list[i] > max:
                max = list[i]
        return max

    def get_maxIndex(self,list):
        max,index = list[0],0
        for i in range(len(list)):
            if list[i] > max:
                max,index = list[i],i
        return index

    def eval(self,encodes,dataset_name,parameter):
        cor =0
        total= 0
        for e in encodes:
            sum = np.array([0.0 for i in range(parameter.class_num)])
            for cl in e.code_list:
                sum+=self.sum_weight_matrix[cl]
            if self.get_maxIndex(sum) == self.get_maxIndex(e.label):
                cor+=1
            total+=1
        if dataset_name=='dev' and cor/total > 0.4:
            print('*******')
        print(dataset_name+' accuracy:', cor/total)
        return cor/total

    def encode_random(self,o_encodes):
        index_list = []
        for i in range(len(o_encodes)):
            index_list.append(i)
        random.seed(200)
        random.shuffle(index_list)
        n_encodes = []
        for i in index_list:
            encode = Encode()
            encode.code_list = o_encodes[i].code_list
            encode.label = o_encodes[i].label
            n_encodes.append(encode)
        return n_encodes