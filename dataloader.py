

from os import environ
from os.path import join
from glob import glob
from pyarrow.parquet import ParquetFile
from random import shuffle


BASE_DIR = environ['BASE_DIR']

def loader(k=1):
    assert k == 1

    parent_path = join(BASE_DIR, 'base_data_climbmix')
    for path in glob(join(parent_path, '*.parquet')):
        f = ParquetFile(path)
        row_nums = list(range(f.num_row_groups))
        shuffle(row_nums)
        for r in row_nums:
            row_group = f.read_row_group(r)
            text_list = row_group.column('text').to_pylist()
            for s in text_list:
                yield s
