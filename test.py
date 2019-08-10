import re
import ujson
import ast
from utils import data_utils, com_utils
import pandas
import config

fileConfig = config.FileConfig()

# print(data_utils.find_str_in_brackets('[《大思想的神奇》[pdf] 资料下载] - 妙学巧记'))
#
# pattern_0 = r'\([^()]*\)'  # depth 0 pattern
# pat_left = r'\((?:[^()]|'
# pat_right = r')*\)'
#
#
# def pattern_generate(pattern, depth=0):
#     while (depth):
#         pattern = pat_left + pattern + pat_right
#         depth -= 1
#     return pattern
#
#
# p1 = re.compile(r'[[](.*)[]]', re.S)
# print(re.findall(p1, '[《大思想的神奇》[pdf] 资料下载] - 妙学巧记'))

import thulac

# import pkuseg

#
#
# # cut_client = thulac.thulac(seg_only=True)
# pk_client = pkuseg.pkuseg(user_dict=fileConfig.dir_jieba + fileConfig.file_jieba_dict)
# # print(cut_client.fast_cut('[《大思想的神奇》[pdf] 资料下载] - 妙学巧记', text=True))
# print(pk_client.cut("《发际红》全集百度影音,优酷土豆网在线观看 - 海外..."))


# alia_kb_df = pandas.read_csv(fileConfig.dir_kb_split + fileConfig.file_kb_pandas_alias_split.format(1))
# test = data_utils.pandas_query_vague(alia_kb_df, 'subject', '冰风谷')
# print('test')