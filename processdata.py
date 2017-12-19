import re
import pickle
import os
import random
from nltk.parse import stanford
from nltk.tree import Tree
import time

random.seed(5)
class Instance:
    def __init__(self):
        self.sentence = ''
        self.label = -1

    def show(self):
        print(self.sentence,' ',self.label)

class Code:
    def __init__(self):
        self.code_list = []
        self.label = []

    def show(self):
        print(self.code_list,' ',self.label)

class Graph:
    def __init__(self):
        self.triples = []

    def show(self):
        print(self.triples)

class Feature:
    def __init__(self):
        self.sentence_feat = []
        self.label_feat = ''

    def show(self):
        print(self.sentence_feat,' ',self.label_feat)

class Read_data:
    def __init__(self):
        self.result = []
        # self.Feature = []

    def clean_str(self,string):
        """
                Tokenization/string cleaning for all datasets except for SST.
                Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def process_file(self,path,clean_switch):
        result = []
        with open(path,'r') as f:
            for o in f.readlines():
                info = o.strip().split('|||')
                inst = Instance()
                if clean_switch:
                    inst.sentence = self.clean_str(info[0])
                else:
                    inst.sentence = info[0]
                inst.label = self.clean_str(info[1])
                result.append(inst)
        self.result =  result

    def create_ngram_dict(self,filename):
        if os.path.exists(filename):
            return pickle.load(open(filename,'rb'))
        else:
            ngram_list = []
            for r in self.result:
                s = r.sentence.split(' ')
                for i in range(len(s)):
                    ngram_list.append('unigram='+s[i])
                for i in range(len(s)-1):
                    ngram_list.append('bigram='+s[i]+'#'+s[i+1])
                for i in range(len(s)-2):
                    ngram_list.append('trigram='+s[i]+'#'+s[i+1]+'#'+s[i+2])
            ngram_dict = {}
            ngram_list = list(set(ngram_list))
            for i in range(len(ngram_list)):
                ngram_dict[ngram_list[i]] = i
            ngram_dict['-unknown-']=len(ngram_list)
            pickle.dump(ngram_dict,open(filename,'wb'))
            return ngram_dict

    def create_tree_list(self,model_path,tree_height):
        parser = stanford.StanfordParser(model_path=model_path)
        tree_list = []
        tree_features = []
        for i,r in enumerate(self.result):
            feat = Feature()
            tree = parser.raw_parse(r.sentence)
            print("第",i,"sentence")
            count = 0
            for t in tree:
                for s in t.subtrees(lambda t:t.height()==tree_height):
                    tree_list.append(str(s))
                    feat.sentence_feat.append(str(s))
                    count+=1
            print(count,"--------------------------")
            feat.label_feat=r.label
            tree_features.append(feat)
        print(len(tree_list))
        return tree_list,tree_features

    def extract_dependency_graph(self,model_path,graph_dir,datasetname):
        if os.path.exists(graph_dir+"/"+datasetname+".pkl"):
            return pickle.load(open(graph_dir+"/"+datasetname+".pkl",'rb'))
        else:
            parser = stanford.StanfordDependencyParser(model_path = model_path)
            graph_list = []
            for i,r in enumerate(self.result):
                print(i,'-')
                graph = parser.parse(r.sentence.split(' '))
                triples = []
                for g in graph:
                    for t in g.triples():
                        triples.append(str(t))
                graph_list.append(triples)
            pickle.dump(graph_list,open(graph_dir+"/"+datasetname+".pkl",'wb'))
            return graph_list

    def create_dependency_graph_list(self,model_path,graph_dir,datasetname):
        dependency_list = self.extract_dependency_graph(model_path,graph_dir,datasetname)
        graph_features = []
        graph_list = []
        for i,r in enumerate(self.result):
            feat = Feature()
            triples = dependency_list[i]
            for t in triples:
                strs = str(re.sub('[()\s\']','',t)).split(',')
                #str1:bow+rel
                clean_str1 = "#children="+strs[3]+"#father="+strs[0]+"#relation="+strs[2]
                #str2:bow+rel+tag
                # clean_str2 = "#children="+strs[3]+"#tag="+strs[4]+"#father="+strs[0]+"#tag="+strs[1]+"#relation="+strs[2]
                #str3:bow
                # clean_str3 = "#children="+strs[3]+"#father="+strs[0]
                #str4:bow+tag
                # clean_str4 = "#children="+strs[3]+"#tag="+strs[4]+"#father="+strs[0]+"#tag="+strs[1]

                graph_list.append(clean_str1)
                # graph_list.append(clean_str2)
                # graph_list.append(clean_str3)
                # graph_list.append(clean_str4)

                # feat.sentence_feat.append(clean_str1)
                # feat.sentence_feat.append(clean_str2)
                # feat.sentence_feat.append(clean_str3)
                # feat.sentence_feat.append(clean_str4)

            feat.label_feat = r.label
            graph_features.append(feat)
        # print(len(graph_list))
        return graph_list,graph_features

    def create_ngram_list(self):
        ngram_list = []
        ngram_features = []
        for r in self.result:
            s = r.sentence.split(' ')
            feat = Feature()
            for i in range(len(s)):
                ngram_list.append('unigram=' + s[i])
                feat.sentence_feat.append('unigram=' + s[i])
            for i in range(len(s) - 1):
                ngram_list.append('bigram=' + s[i] + '#' + s[i + 1])
                feat.sentence_feat.append('bigram=' + s[i] + '#' + s[i + 1])
            for i in range(len(s) - 2):
                ngram_list.append('trigram=' + s[i] + '#' + s[i + 1] + '#' + s[i + 2])
                feat.sentence_feat.append('trigram=' + s[i] + '#' + s[i + 1] + '#' + s[i + 2])
            feat.label_feat=r.label
            ngram_features.append(feat)
        return ngram_list,ngram_features

    def create_feature(self,parameter,dict_switch,datasetname):
        print("dataset: "+datasetname+"-------------")
        feature_dict = {}
        features = []
        feature_list = []

        # tree_list, tree_features = self.create_tree_list(parameter.model_path, parameter.tree_height)
        graph_list,graph_features = self.create_dependency_graph_list(parameter.model_path,parameter.graph_dir,datasetname)
        ngram_list,ngram_features = self.create_ngram_list()

        for i,r in enumerate(self.result):
            feat = Feature()
            # feat.sentence_feat.extend(tree_features[i].sentence_feat)
            feat.sentence_feat.extend(graph_features[i].sentence_feat)
            feat.sentence_feat.extend(ngram_features[i].sentence_feat)
            feat.label_feat=r.label
            features.append(feat)
        if dict_switch:
            feature_list.extend(graph_list)
            feature_list.extend(ngram_list)
            dict_list = list(set(feature_list))
            for i in range(len(dict_list)):
                feature_dict[dict_list[i]] = i
            feature_dict['-unknown-'] = len(dict_list)
        else:
            pass
        return feature_dict,features

    def remove_duplicate(self,input):
        output = []
        for i in input:
            if i not in output:
                output.append(i)
            else:
                pass
        return output

    def create_freq_dict(self,threshold,ngram_dict):
        ngram_list = []
        for r in self.result:
            s = r.sentence.split(' ')
            for i in range(len(s)):
                ngram_list.append('unigram=' + s[i])
            for i in range(len(s) - 1):
                ngram_list.append('bigram=' + s[i] + '#' + s[i + 1])
            for i in range(len(s) - 2):
                ngram_list.append('trigram=' + s[i] + '#' + s[i + 1] + '#' + s[i + 2])
        freq_dict = {}
        for nl in ngram_list:
            if nl not in freq_dict.keys():
                freq_dict[nl] = 0
            else:
                freq_dict[nl] += 1
        delete_key = []
        for nd in ngram_dict.keys():
            if nd == '-unknown-':
                continue
            if freq_dict[nd] < threshold:
                delete_key.append(nd)
        for dk in delete_key:
            ngram_dict.pop(dk)
        i=0
        for k in ngram_dict.keys():
            ngram_dict[k] = i
            i+=1
        return ngram_dict

class Encode:
    def __init__(self):
        pass

    def encode(self,data_result,dict):
        encodes = []
        for r in data_result:
            e = Code()
            s = r.sentence.split(' ')
            for i in range(len(s)):
                if 'unigram='+s[i] in dict.keys():
                    e.code_list.append(dict['unigram='+s[i]])
                else:
                    e.code_list.append(dict['-unknown-'])
            for i in range(len(s)-1):
                if 'bigram='+s[i]+'#'+s[i+1] in dict.keys():
                    e.code_list.append(dict['bigram='+s[i]+'#'+s[i+1]])
                else:
                    e.code_list.append(dict['-unknown-'])
            for i in range(len(s)-2):
                if 'trigram='+s[i]+'#'+s[i+1]+'#'+s[i+2] in dict.keys():
                    e.code_list.append(dict['trigram='+s[i]+'#'+s[i+1]+'#'+s[i+2]])
                else:
                    e.code_list.append(dict['-unknown-'])
            if r.label == '0':
                e.label = [1,0,0,0,0]
            elif r.label == '1':
                e.label = [0,1,0,0,0]
            elif r.label == '2':
                e.label = [0,0,1,0,0]
            elif r.label == '3':
                e.label = [0,0,0,1,0]
            elif r.label == '4':
                e.label = [0,0,0,0,1]
            encodes.append(e)
        return encodes

    def feat_encode(self,features,dictionary):
        encodes = []
        assert len(features) > 0
        for feat in features:
            e = Code()
            for sf in feat.sentence_feat:
                if sf in dictionary.keys():
                    e.code_list.append(dictionary[sf])
                else:
                    # pass
                    e.code_list.append(dictionary['-unknown-'])
            if feat.label_feat == '0':
                e.label = [1, 0, 0, 0, 0]
            elif feat.label_feat == '1':
                e.label = [0, 1, 0, 0, 0]
            elif feat.label_feat == '2':
                e.label = [0, 0, 1, 0, 0]
            elif feat.label_feat == '3':
                e.label = [0, 0, 0, 1, 0]
            elif feat.label_feat == '4':
                e.label = [0, 0, 0, 0, 1]
            encodes.append(e)
        return encodes

        # parser = stanford.StanfordParser(model_path=parameter.model_path)
        # if os.path.exists(datafile):
        #     return pickle.load(open(datafile,"rb"))
        # else:
        #     for i,r in enumerate(data_result):
        #         print("第",i+1)
        #         e = Code()
        #         f = Feature()
        #         tree = parser.raw_parse(r.sentence)
        #         for t in tree:
        #             for s in t.subtrees(lambda t: t.height() == parameter.tree_height):
        #                 f.sentence_feat.append(str(s))
        #                 if str(s) in dict.keys():
        #                     e.code_list.append(dict[str(s)])
        #                 else:
        #                     e.code_list.append(dict['-unknown-'])
        #         s = r.sentence.split(' ')
        #         for i in range(len(s)):
        #             f.sentence_feat.append('unigram=' + s[i])
        #             if 'unigram=' + s[i] in dict.keys():
        #                 e.code_list.append(dict['unigram=' + s[i]])
        #             else:
        #                 e.code_list.append(dict['-unknown-'])
        #         for i in range(len(s) - 1):
        #             f.sentence_feat.append('bigram=' + s[i] + '#' + s[i + 1])
        #             if 'bigram=' + s[i] + '#' + s[i + 1] in dict.keys():
        #                 e.code_list.append(dict['bigram=' + s[i] + '#' + s[i + 1]])
        #             else:
        #                 e.code_list.append(dict['-unknown-'])
        #         for i in range(len(s) - 2):
        #             f.sentence_feat.append('trigram=' + s[i] + '#' + s[i + 1] + '#' + s[i + 2])
        #             if 'trigram=' + s[i] + '#' + s[i + 1] + '#' + s[i + 2] in dict.keys():
        #                 e.code_list.append(dict['trigram=' + s[i] + '#' + s[i + 1] + '#' + s[i + 2]])
        #             else:
        #                 e.code_list.append(dict['-unknown-'])
        #         f.label_feat.append(r.label)
        #         if r.label == '0':
        #             e.label = [1, 0, 0, 0, 0]
        #         elif r.label == '1':
        #             e.label = [0, 1, 0, 0, 0]
        #         elif r.label == '2':
        #             e.label = [0, 0, 1, 0, 0]
        #         elif r.label == '3':
        #             e.label = [0, 0, 0, 1, 0]
        #         elif r.label == '4':
        #             e.label = [0, 0, 0, 0, 1]
        #         encodes.append(e)
        #         print("-------------------")
        #     pickle.dump(encodes,open(datafile,'wb'))
        #     pickle.dump(features,open(featfile,"wb"))
        #     return encodes