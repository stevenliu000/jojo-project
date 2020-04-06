import os
from concurrent.futures import ProcessPoolExecutor
import edge_promoting_thread
root = os.path.join('/home/stevenliu/jojo-project/baseline/data/tgt', 'train')
save = os.path.join('/home/stevenliu', 'pair')
cores = 5
def main():
    file_list = os.listdir(root)
    num_data = len(file_list)
    indics = int(num_data/cores)
    edge_promoting_thread.edge_promoting(root, file_list, save)
    with ProcessPoolExecutor(max_workers=cores) as executor:
    for i in range(cores):
        executor.submit(edge_promoting_thread.edge_promoting, root, file_list[i*indics:(i+1)*indics], save)

