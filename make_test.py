import csv
import numpy as np
from tqdm import tqdm
import sys
from itertools import permutations

max_num = int(sys.argv[1])

read_file = "./Dataset/STS-B/old_test.tsv"

f = open(read_file)
tsv_data = csv.reader(f, delimiter="\t")

list_of_data = []
for i, row in enumerate(tsv_data, 1):
    list_of_data.append(row)

f.close()

# get sentences
sentences1 = []
sentences2 = []
for i in tqdm(range(len(list_of_data))):
    try:
        sentences1.append(list_of_data[i][7])
    except:
        pass
    try:
        sentences2.append(list_of_data[i][8])
    except:
        pass

# merge lists
#sentences1 = sentences1[1:len(sentences1)-1]
#sentences2 = sentences2[1:len(sentences2)-1]
sentences_all = sentences1+sentences2
sentences1 = sentences_all[1:max_num]
sentences2 = sentences_all[1:max_num]
merged_list = []
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        merged_list.append([sentences1[i], sentences2[j]])

#"""
print(list_of_data[0])

index = list(range(len(merged_list)))
genre = []
filename = []
year = []
source1 = []
source2 = []
for i in range(len(merged_list)):
    genre.append("misc")
    filename.append("archan")
    year.append("2020")
    source1.append("none")
    source2.append("none")

old_index = index

with open("new_test.tsv", "w", newline='') as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(list_of_data[0])
    for i in tqdm(range(len(merged_list))):
        writer.writerow([index[i], genre[i], filename[i], year[i], old_index[i], source1[i], source2[i], merged_list[i][0], merged_list[i][1]])

#"""
print(len(sentences1), len(sentences2))
