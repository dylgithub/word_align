# -*- coding: utf-8 -*-
import jieba

data_list = []
with open("word_align_use.txt", "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        new_line = line.strip()
        data1, data2 = new_line.split(" ||| ")
        data_list.append(" ".join(jieba.lcut(data1)) + " ||| " + " ".join(jieba.lcut(data2)))
with open("word_align_use_final.txt", "w", encoding="utf-8") as fw:
    for data in data_list:
        fw.write(data + "\n")
