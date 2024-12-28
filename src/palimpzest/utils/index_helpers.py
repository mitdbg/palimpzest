import re


def get_index_str(index):
    regex = re.compile('<(.*?) object at.*?>')
    return regex.match(str(index))[1]
