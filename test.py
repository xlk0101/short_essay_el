import re
import ujson
import ast
from utils import data_utils

print(data_utils.find_str_in_brackets('[《大思想的神奇》[pdf] 资料下载] - 妙学巧记'))

pattern_0 = r'\([^()]*\)'  # depth 0 pattern
pat_left = r'\((?:[^()]|'
pat_right = r')*\)'


def pattern_generate(pattern, depth=0):
    while (depth):
        pattern = pat_left + pattern + pat_right
        depth -= 1
    return pattern


p1 = re.compile(r'[[](.*)[]]', re.S)
print(re.findall(p1, '[《大思想的神奇》[pdf] 资料下载] - 妙学巧记'))
