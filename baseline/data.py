from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import edge_promoting_thread
import os
from pathlib import Path
import shutil

cwd = os.getcwd()
'''
print('start move src data')
# move src data
src_path = os.path.join(cwd, "data/land/landscape-pictures")
src_files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
src_files.sort()
train_files = src_files[:4000]

train_path = os.path.join(cwd, './data/src/train')
if not os.path.isdir(train_path):
    os.makedirs(train_path)
    
for i in train_files:
    shutil.copyfile(os.path.join(src_path, i), (os.path.join(train_path, i)))

testfiles = src_files[4000:]
print(len(src_files))
test_path = os.path.join(cwd, 'data/src/test')
if not os.path.isdir(test_path):
    os.makedirs(test_path)
for i in testfiles:
    shutil.copyfile(os.path.join(src_path, i), (os.path.join(test_path, i)))
'''
print('start move tgt data')
# move tgt data
jojo_path = os.path.join(cwd, 'data/peper/images/Kimi_No_Na_Wa')

jojo_filenames = []


jojo_filenames = [f for f in os.listdir(jojo_path) if os.path.isfile(os.path.join(jojo_path, f))]
'''
for s in os.listdir(jojo_path):
    if os.path.isdir(os.path.join(jojo_path, s)):
        subdir = os.path.join(jojo_path, s)
        for f in os.listdir(subdir):
            filepath = os.path.join(subdir, f)
            if os.path.isfile(filepath):
                jojo_filenames.append(filepath)
'''
jojo_filenames.sort()

tgt_path = os.path.join(cwd, 'data/name/train')
if not os.path.isdir(tgt_path):
    os.makedirs(tgt_path)
j = 0
for i in jojo_filenames[:]: 
    shutil.copyfile(os.path.join(jojo_path, i), os.path.join(tgt_path, str(j)+'.jpg'))
    j += 1

print('edge-promoting start!!')
# edge_promoting
root = './data/name/train'
save = os.path.join('./data/name', 'pair')
if not os.path.isdir(save):
    os.makedirs(save)
cores = os.cpu_count()

file_list = os.listdir(root)
num_data = len(file_list)
indics = int(num_data/cores)
with ProcessPoolExecutor(max_workers=cores) as executor:
    for i in range(cores):
        executor.submit(edge_promoting_thread.edge_promoting, root, file_list[i*indics:(i+1)*indics], save)

