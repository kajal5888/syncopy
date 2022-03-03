import os
from syncopy.shared.errors import SPYIOError
from syncopy.shared.parsers import io_parser

mat_file_dir = '/cs/scratch/syncopy/MAT-Files'
mat_name = 'matdata-v73.mat'
fname = os.path.join(mat_file_dir, mat_name)

print(f'os.path finds {mat_file_dir} - {os.path.exists(mat_file_dir)}')
print(f'os.path finds {fname} - {os.path.exists(fname)}')
try:
    io_parser(fname, isfile=True)
except SPYIOError as err:
    print(err)
