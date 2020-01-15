# coding=utf-8
import io
import json
import collections
import tensorflow as tf
from bert import bert_tokenization as tokenization

# get genres for each movie
movie_titles_metadata = open('movie_titles_metadata.txt', "r").readlines()
movie_genres_dict = {}
for line in movie_titles_metadata:
    splitline = line.split(' +++$+++ ')
    movie_id = splitline[0]
    genre = splitline[-1].strip()
    movie_genres_dict[movie_id] = genre
    #print(movie_id + " "  + genre)

# get dict for charactors
movie_characters_metadata = open('movie_characters_metadata.txt', "r").readlines()
characters_id2name_dict = {}
characters_id2sex_dict = collections.defaultdict(lambda: '?')
for line in movie_characters_metadata:
    splitline = line.split(' +++$+++ ')
    chara_id = splitline[0]
    chara_name = splitline[1]
    chara_sex = splitline[-2]
    if chara_sex == 'm' or chara_sex == 'f':
        characters_id2name_dict[chara_id] = chara_name
        characters_id2sex_dict[chara_id] = chara_sex
        #print(chara_id + ' ' + chara_name + ' ' + chara_sex)

#get lines in movies
#out_file = open('standard_lines.txt', 'w')
'''
out_file_for_bert = open('movie_lines.tsv' , 'w')
out_file_for_bert.write('\xEF\xBB\xBF')
count_down_male = 50000
count_down_female = 50000
'''

lines_id2charaid_dict = collections.defaultdict(lambda: '?')
lines_id2line_dict = collections.defaultdict(lambda: '?')
#out_file.write('\xEF\xBB\xBF') # using UTF-8
movie_lines = open('movie_lines.txt', 'r').readlines()
for line in movie_lines:
    splitline = line.split(' +++$+++ ')
    line_id = splitline[0]
    chara_id = splitline[1]
    if not chara_id in characters_id2name_dict.keys():
        continue
    movie_id = splitline[2]
    sentence = splitline[-1].strip()
    #print(sentence)
    chara_sex = characters_id2sex_dict[chara_id]
    sex_num = '0' if chara_sex == 'm' else '1'  # 0 is male, 1 is female
    lines_id2charaid_dict[line_id] = chara_id
    lines_id2line_dict[line_id] = sentence

    '''
    if chara_sex == 'm' and count_down_male > 0:
        count_down_male -= 1
        out_file_for_bert.write(chara_sex + '\t' + sentence + '\n')
        #out_file.write(line_id + '\t' + chara_id + '\t' + characters_id2name_dict[chara_id] + '\t' + movie_genres_dict[movie_id] + '\t' + sentence + '\t' + sex_num + '\n')
    elif chara_sex == 'f' and count_down_female > 0:
        count_down_female -= 1
        out_file_for_bert.write(chara_sex + '\t' + sentence + '\n')
        #out_file.write(line_id + '\t' + chara_id + '\t' + characters_id2name_dict[chara_id] + '\t' + movie_genres_dict[movie_id] + '\t' + sentence + '\t' + sex_num + '\n')
#out_file.close()
    else:
        out_test_file = open('movie_lines.txt')
    '''


#out_file_for_bert.close()
#out = open('test.txt', 'w')
#for key in lines_id2charaid_dict.keys():
#    out.write(key)
#out.close()
#out_female2male_conver = open('female2male.txt', 'w')


def check(sentence, max_seq_length = 128):
    tokens =  ["[CLS]"] + tokenizer.tokenize(line) + ["[SEP]"]
    return True if len(tokens) <=- 128 and len(tokens) > 2 else False

out_female2male_conver_part1 =  open('female2male_part1.txt', 'w')
out_female2male_conver_part2 =  open('female2male_part2.txt', 'w')
movie_conversations = open('movie_conversations.txt', 'r').readlines()
for line in movie_conversations:
    splitline = line.split(' +++$+++ ')
    chara_id1 = splitline[0]
    chara_id2 = splitline[1]
    conversation_sentence_list_str = splitline[-1].strip()
    #print(conversation_sentence_list_str)
    conversation_sentence_list = [x.strip().replace('\'', '') for x in conversation_sentence_list_str.replace('[','').replace(']','').split(',')]
    #print(conversation_sentence_list)
    for sentence_id in range(0, len(conversation_sentence_list) - 1):
        lineid1 = conversation_sentence_list[sentence_id]
        lineid2 = conversation_sentence_list[sentence_id + 1]
        #print(lineid1 + "\t" + lineid2)
        #print(lines_id2charaid_dict[lineid1])
        #print(characters_id2sex_dict[lines_id2charaid_dict[lineid1]])
        if characters_id2sex_dict[lines_id2charaid_dict[lineid1]] == 'f' and characters_id2sex_dict[lines_id2charaid_dict[lineid2]] == 'm':
            #out_female2male_conver.write(lines_id2line_dict[lineid1] + '\t' + lines_id2line_dict[lineid2] + '\n')
            sentence1 = lines_id2line_dict[lineid1]
            if check(sentence1) == false:
                continue
            sentence2 = lines_id2line_dict[lineid2]
            if check(sentence2) == flase:
                continue
            out_female2male_conver_part1.write(lines_id2line_dict[lineid1] + '\n')
            out_female2male_conver_part2.write(lines_id2line_dict[lineid2] + '\n')
out_female2male_conver_part1.close()
out_female2male_conver_part2.close()