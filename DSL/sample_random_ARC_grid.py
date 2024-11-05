import random 
import os
import json
import threading
import multiprocessing
import pickle
from tqdm import tqdm

RE_ARC_LOCATION = r"C:\Users\Francesco\Desktop\github_repos\ARC\code\auxillary_github_repos\re-arc\re_arc\re_arc\tasks\\"

file_names = os.listdir(RE_ARC_LOCATION)
number_of_files = len(file_names)
data_dictionary = {}

from time import perf_counter

# def get_all_data():    
#     for file_name in file_names:
#         fi = open(RE_ARC_LOCATION+file_name)
#         file_data = json.load(fi)
#         fi.close()
#         print(file_name[:-5])
# data_dictionaries = [{},{},{},{}]
def get_a_quarter_data(i, q):    
    data={}
    n = 100
    for file_name in tqdm(file_names[i*n:(i+1)*n]):
        fi = open(RE_ARC_LOCATION+file_name)
        data[file_name[:-5]] = json.load(fi)
        print(file_name[:-5])
        fi.close()
    print(f"process {i} ended!")
    q.put(data)
def get_a_little_data(i, q, quarter_size):    
    data={}
    n = quarter_size
    for file_name in tqdm(file_names[i*n:(i+1)*n]):
        fi = open(RE_ARC_LOCATION+file_name)
        data[file_name[:-5]] = json.load(fi)
        print(file_name[:-5])
        fi.close()
    print(f"process {i} ended!")
    q.put(data)

def pickle_all_data():
    st = perf_counter()
    processes = []
    q = multiprocessing.Queue()
    for i in range(4):
        processes.append(process:=multiprocessing.Process(None, get_a_quarter_data, args = (i,q)))
        process.start()
    data_dictionary.update(q.get())
    data_dictionary.update(q.get())
    data_dictionary.update(q.get())
    data_dictionary.update(q.get())
    [process.join() for process in processes]
    print(len(data_dictionary))
    pickle.dump(data_dictionary, open("re_arc_total.pkl", "wb"))
    en = perf_counter()
    print(en-st)
def pickle_some_data(size: int):
    assert size < 400 and size > 0
    st = perf_counter()
    processes = []
    q = multiprocessing.Queue()
    for i in range(4):
        processes.append(process:=multiprocessing.Process(None, get_a_little_data, args = (i,q,size//4)))
        process.start()
    data_dictionary.update(q.get())
    data_dictionary.update(q.get())
    data_dictionary.update(q.get())
    data_dictionary.update(q.get())
    [process.join() for process in processes]
    print(len(data_dictionary))
    pickle.dump(data_dictionary, open(f"re_arc_partial_{size}.pkl", "wb"))
    en = perf_counter()
    print(en-st)

def get_pickle_data(pickle_file_name: str = "re_arc_total.pkl"):
    print(f"Loading pickled data from {pickle_file_name}")
    return pickle.load(open(pickle_file_name,"rb"))

def sample_random_ARC_grid_from(data, data_keys, num_files=400, len_file=1000):
    random_file = random.randint(0,num_files-1)
    random_io = random.randint(0,len_file-1)
    io = data[data_keys[random_file]][random_io]
    return io

st = perf_counter()
# pickle_all_data()
# pickle_some_data(40)
ARC_sampling_data: dict = get_pickle_data("re_arc_partial_40.pkl")
# ARC_sampling_data: dict = get_pickle_data()
print(len(ARC_sampling_data))
print(len(ARC_sampling_data.keys()))
en = perf_counter()
print(en-st)

def sample_random_ARC_grid_full():
    return sample_random_ARC_grid_from(ARC_sampling_data, list(ARC_sampling_data.keys()))
def sample_random_ARC_grid_partial(size):
    return sample_random_ARC_grid_from(ARC_sampling_data, list(ARC_sampling_data.keys()), num_files=size)

# st = perf_counter()
# for i in range(100):
#     io = sample_random_ARC_grid(data, list(data.keys()))
#     inp = io["input"]
#     out = io["output"]
# en = perf_counter()
# print(inp)
# print(out)
# print(en-st)

if __name__ == '__main__':
    # st = perf_counter()
    # # pickle_all_data()
    # pickle_some_data(40)
    # ARC_sampling_data: dict = get_pickle_data("re_arc_partial_40.pkl")
    # # ARC_sampling_data: dict = get_pickle_data()
    # print(len(ARC_sampling_data))
    # print(len(ARC_sampling_data.keys()))
    # en = perf_counter()
    # print(en-st)
    pass