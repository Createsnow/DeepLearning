# -*- coding:utf-8 -*-

"""
生成评估,测试,训练的数据集（命名实体识别）
1：2：13
"""

import os
import csv
import jieba
import jieba.posseg as pseg

c_root = os.getcwd() + os.sep + "source_file" + os.sep

dev = open("data\example.dev", 'w', encoding='utf8')
test = open("data\example.test", 'w', encoding='utf8')
train = open("data\example.train", 'w', encoding='utf8')
#词性
biaoji = set(['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW', 'CL'])

fuhao = set(['；', '。', '?', '？', '!', '！', ';'])

#迭代读取（按行读取）【基本open读取都是迭代读取，一直读取到全文】
dics = csv.reader(open("DICT_NOW.csv", 'r', encoding='utf8'))
for row in dics:
    if len(row) == 2:
        jieba.add_word(row[0].strip(), tag=row[1].strip())

split_num = 0
for file in os.listdir(c_root):
    if "txtoriginal.txt" in file:
        fp = open(c_root + file, 'r', encoding="utf8")
        for line in fp:
            split_num += 1
            print(line)
            words = pseg.cut(line)
            print(words)
            # key:表示词，value：表示词性也就是标签######################################
            for key, value in words:
                print(key)
                print(value)
                if value.strip() and key.strip():
                    # if split_num % 15 < 2:
                    #     index = str(1)
                    # elif split_num % 15 > 1 and split_num % 15 < 4:
                    #     index = str(2)
                    # else:
                    #     index = str(3)
                    #优化
                    index=str(1) if split_num % 15 < 2 else str(2) if split_num % 15 > 1 and split_num % 15 < 4 else str(3)
                    if value not in biaoji:
                        value = 'O'
                        for achar in key.strip():
                            if achar and achar.strip() in fuhao:
                                string = achar + " " + value.strip() + "\n" + "\n"
                                # if index == '1':
                                #     dev.write(string)
                                # elif index == '2':
                                #     test.write(string)
                                # elif index == '3':
                                #     train.write(string)
                                # else:
                                #     pass
                                #优化
                                dev.write(string) if index == '1' else test.write(string) if index == '2' else train.write(string)

                            elif achar.strip() and achar.strip() not in fuhao:
                                string = achar + " " + value.strip() + "\n"
                                # if index == '1':
                                #     dev.write(string)
                                # elif index == '2':
                                #     test.write(string)
                                # elif index == '3':
                                #     train.write(string)
                                # else:
                                #     pass
                                #优化
                                dev.write(string) if index == '1' else test.write(string) if index == '2' else train.write(string)
                            else:
                                continue

                    elif value.strip() in biaoji:
                        begin = 0
                        for char in key.strip():
                            if begin == 0:
                                begin += 1
                                string1 = char + ' ' + 'B-' + value.strip() + '\n'
                                # if index == '1':
                                #     dev.write(string1)
                                # elif index == '2':
                                #     test.write(string1)
                                # elif index == '3':
                                #     train.write(string1)
                                # else:
                                #     pass
                                #优化
                                dev.write(string1) if index == '1' else test.write(string1) if index == '2' else train.write(string1)
                            else:
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                # if index == '1':
                                #     dev.write(string1)
                                # elif index == '2':
                                #     test.write(string1)
                                # elif index == '3':
                                #     train.write(string1)
                                # else:
                                #     pass
                                #优化
                                dev.write(string1) if index == '1' else test.write(string1) if index == '2' else train.write(string1)
                    else:
                        continue
dev.close()
test.close()
train.close()
