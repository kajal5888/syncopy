import os
from syncopy.shared.errors import SPYIOError
from syncopy.shared.parsers import io_parser

mat_file_dir = '/cs/scratch/syncopy/MAT-Files'
mat_name = 'matdata-v73.mat'
fname = os.path.join(mat_file_dir, mat_name)

def test_os_access():
    print(f'os.path finds {mat_file_dir} - {os.path.exists(mat_file_dir)}')
    print(f'os.path finds {fname} - {os.path.exists(fname)}')
    assert os.path.exists(mat_file_dir)
    assert os.path.exists(fname)

def test_spyio_parser():
    io_parser(fname, isfile=True)
