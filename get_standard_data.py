# coding=utf-8
import io

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
characters_id2sex_dict = {}
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
out_file = open('standard_lines.txt', 'w')
count_down_male = 50000
count_down_female = 50000
out_file.write('\xEF\xBB\xBF') # using UTF-8
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
    if chara_sex == 'm' and count_down_male > 0:
        count_down_male -= 1
        out_file.write(line_id + '\t' + chara_id + '\t' + characters_id2name_dict[chara_id] + '\t' + movie_genres_dict[movie_id] + '\t' + sentence + '\t' + sex_num + '\n')
    elif chara_sex == 'f' and count_down_female > 0:
        count_down_female -= 1
        out_file.write(line_id + '\t' + chara_id + '\t' + characters_id2name_dict[chara_id] + '\t' + movie_genres_dict[movie_id] + '\t' + sentence + '\t' + sex_num + '\n')
out_file.close()