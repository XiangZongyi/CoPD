
# -*- Encoding:UTF-8 -*-

# raw Douban data --> dataset Movei-Book and Movie-Music

import numpy as np
import os

class DataSet(object):
    def __init__(self, fileNameA, fileNameB):
        self.dataA, self.dataB = self.getData(fileNameA, fileNameB)
        self.trainA, self.test_originA, item_in_trainA = self.getTrainTest(domain='A')
        self.trainB, self.test_originB, item_in_trainB = self.getTrainTest(domain='B')
        # self.trainDict = self.getTrainDict()
        self.testA = self.filterColdInTest(item_in_trainA, domain='A')
        self.testB = self.filterColdInTest(item_in_trainB, domain='B')

    def getData(self, fileNameA, fileNameB):
        user2itemA = dict()
        user2itemB = dict()
        dataA = []
        dataB = []
        
        print("Loading %s data set..."%(fileNameA))
        filePath = './'+fileNameA+'/ratings.dat'
        with open(filePath, 'r') as f:
            for line in f:
                if line:
                    lines = line.split("\t")
                    user = int(lines[0])
                    movie = int(lines[1])
                    if user not in user2itemA:
                        user2itemA[user] = []
                    user2itemA[user] += [movie]
        
        print("Loading %s data set..."%(fileNameB))
        filePath = './'+fileNameB+'/ratings.dat'
        with open(filePath, 'r') as f:
            for line in f:
                if line:
                    lines = line.split("\t")
                    user = int(lines[0])
                    movie = int(lines[1])
                    if user not in user2itemB:
                        user2itemB[user] = []
                    user2itemB[user] += [movie]
        
        user2itemA, user2itemB = self.filterOverlapUser(user2itemA, user2itemB)
        # reindex
        user_map = {}
        item_map = {}
        for user, items in user2itemA.items():
            if user not in user_map.keys():
                user_map[user] = len(user_map)
            for item in items:
                if item not in item_map.keys():
                    item_map[item] = len(item_map)
                dataA.append((user_map[user], item_map[item], 1, 0))
        # reindex
        user_map = {}
        item_map = {}
        for user, items in user2itemB.items():
            if user not in user_map.keys():
                user_map[user] = len(user_map)
            for item in items:
                if item not in item_map.keys():
                    item_map[item] = len(item_map)
                dataB.append((user_map[user], item_map[item], 1, 0))
        return dataA, dataB
    
    def filterOverlapUser(self, user2itemA, user2itemB):
        temp_keys = list(user2itemB.keys())
        for item in temp_keys:
            if item not in user2itemA:
                user2itemB.pop(item)
        temp_keys = list(user2itemA.keys())
        for item in temp_keys:
            if item not in user2itemB:
                user2itemA.pop(item)
        return user2itemA, user2itemB

    def getTrainTest(self, domain='A'):
        if domain == 'A':
            data = self.dataA
        else:
            data = self.dataB
        data = sorted(data, key=lambda x: (x[0], x[3]))

        train = []
        test = []
        item_in_train = set()
        # for i in range(len(data)-1):
        user, item, rate = data[0][0], data[0][1], data[0][2]
        train.append((user, item, rate))
        for i in range(1, len(data)-1):
            user = data[i][0]
            item = data[i][1]
            rate = data[i][2]
            if data[i][0] != data[i+1][0] and data[i-1][0] == data[i][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))
                item_in_train.add(item)
        if data[-1][0] == data[-2][0]:
            test.append((data[-1][0], data[-1][1], data[-1][2]))
        else:
            train.append((data[-1][0], data[-1][1], data[-1][2]))
        return train, test, item_in_train
    
    def filterColdInTest(self, item_in_train, domain='A'):
        if domain == 'A':
            test_origin = self.test_originA
        else:
            test_origin = self.test_originB
        test = []
        for i in range(len(test_origin)):
            user = test_origin[i][0]
            item = test_origin[i][1]
            rate = test_origin[i][2]
            if item in item_in_train:
                test.append((user, item, rate))
        return test
        
def saveFile(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            row_str = '\t'.join(map(str, row))
            f.write(row_str + '\n')
            
if __name__ == '__main__':
    tasks = [['douban_movie', 'douban_book'], ['douban_movie', 'douban_music']]
    for [dataName_A, dataName_B] in tasks:
        dataSet = DataSet(dataName_A, dataName_B)
        
        train_A = dataSet.trainA
        test_A = dataSet.testA
        # test_A_origin = dataSet.test_originA
        train_B = dataSet.trainB
        test_B = dataSet.testB
        # test_B_origin = dataSet.test_originB
        
        save_nameA = dataName_A.split('_')[1] + '_' + dataName_B.split('_')[1]
        save_nameB = dataName_B.split('_')[1] + '_' + dataName_A.split('_')[1]
        save_root_A = '../dataset/' + save_nameA + '/'
        save_root_B = '../dataset/' + save_nameB + '/'
        
        if not os.path.exists(save_root_A):
            os.makedirs(save_root_A)
        saveFile(save_root_A + 'train.txt', train_A)
        saveFile(save_root_A + 'test.txt', test_A)
        # saveFile(save_root_A + 'test_origin.txt', test_A_origin)
        
        if not os.path.exists(save_root_B):
            os.makedirs(save_root_B)
        saveFile(save_root_B + 'train.txt', train_B)
        saveFile(save_root_B + 'test.txt', test_B)
        # saveFile(save_root_B + 'test_origin.txt', test_B_origin)

        
        