import dill as pickle
from torch.utils.data import IterableDataset
from ARC_type_system import Arrow, GRID, FrozenSet
from ARC_dsl_cfg import ARC_DSL
from ARC_cfg_pcfg import ARC_CFG
from ARC_program import format_program_full
from ARC_formatted_dsl import primitive_types
from ARC_pcfg import ARC_PCFG
import json
from tqdm import tqdm
from time import perf_counter
from test_speeds import timeitlogs    
from PIL import Image
import torch
from data import Corpus

from manipulate_ARC_images import ARCcmap, to_pil_image, from_2Dgrid_to_pil_image, from_RGB_tensor_to_pil_image



# ARC_dsl:ARC_DSL =pickle.load(open("DSL\\arc_dsl_8type.pkl", "rb"))
# ARC_cfg_1:ARC_CFG=pickle.load(open("DSL\\arc_1gram_8type.pkl", "rb"))
# ARC_cfg_2:ARC_CFG=pickle.load(open("DSL\\arc_2gram_8type.pkl", "rb"))

# ARC_unif_pcfg: ARC_PCFG = ARC_cfg_1.CFG_to_Uniform_PCFG()
# program_generator = ARC_unif_pcfg.sampling()
def is_rectangular_and_filled(grid):
    xdim=len(grid)
    ydim=len(grid[0])
    for row in grid:
        if not all(row):
            return False
        if not len(row) == ydim:
            return False
    return True
def is_good_arc_program(intput, output, pr):
    if output != None and all(output) and len(output) < 30 and len(output) > 1:
        if isinstance(output[0],int):
            raise TypeError(f"Int found from {pr} with output = {output} and input = {intput}")
        if len(frow:=output[0]) > 1 and len(frow) < 31 and (el:=frow[0]) != None and el > 0:
            if is_rectangular_and_filled(output):
                str_program = str(pr)
                if "var0" in str_program:
                    return True, str_program
    return False, ""
def get_one_random_ARC_walk(ARC_grid: list[list[int]], len_walk: int, program_generator: iter, ARC_dsl:ARC_DSL, masked:bool=False) -> str:#list[tuple[list[list[int]],str]]:
    """Will immediately reformat ARC_grid input to Tuple[Tuple[int]]"""
    good = 0
    count = 0
    if len_walk > 1:
        cur_list = [(ARC_grid, "START")]
        ARC_grid = tuple(tuple(row) for row in ARC_grid)
        cur_list = ["["+str(ARC_grid).replace(" ","")+",'*START*']"]
        out = ARC_grid
        while len(cur_list) < len_walk:
            program = next(program_generator)
            count += 1
            inp = out
            out = program.eval_naive(dsl = ARC_dsl, environment=[inp])
            is_good, str_program = is_good_arc_program(inp, out, program)
            if is_good:
                good +=1
                # cur_list.append((out,str_program))
                cur_list.append("["+str(out).replace(" ","")+",'"+str_program+"']")
        # print(f"good, count, good/count: {good}, {count}, {good/count}")
        # return cur_list
        return "["+ ",".join(cur_list) +"]"
    elif len_walk == 1:
        ARC_grid = tuple(tuple(row) for row in ARC_grid)
        while True:
            program = next(program_generator)
            count += 1
            out = program.eval_naive(dsl = ARC_dsl, environment=[ARC_grid])
            is_good, str_program = is_good_arc_program(ARC_grid, out, program)
            if is_good:
                good +=1
                break     
        # print(f"good, count, good/count: {good}, {count}, {good/count}")
        # return cur_list
        # return f"[{str(ARC_grid).replace(" ","")},{str_program},{str(out).replace(" ","")}]"
        if not masked:
            return (ARC_grid, str_program, program, out)
        else:
            masked_str_program = mask_program(str_program, has_expansion = True, returns_only_one_masked_str = True)
            return (ARC_grid, str_program, masked_str_program, program, out)
def format_str_program(pr_str:str)->str:
    """E.g. (power (branch F NINE) NEG_TWO FIVE)  to  ( power ( branch F NINE ) NEG_TWO FIVE )"""
    return pr_str.replace("(", "( ").replace(")", " )").strip()
def get_one_random_ARC_program(ARC_grid: list[list[int]], program_generator: iter, ARC_dsl:ARC_DSL) -> tuple[str,str]:#list[tuple[list[list[int]],str]]:
    """Will immediately reformat ARC_grid input to Tuple[Tuple[int]]"""
    count = 0
    ARC_grid = tuple(tuple(row) for row in ARC_grid)
    while True:
        program = next(program_generator)
        count += 1
        out = program.eval_naive(dsl = ARC_dsl, environment=[ARC_grid])
        is_good, str_program = is_good_arc_program(ARC_grid, out, program)
        if is_good:
            #formatted_pr_str = format_str_program(str_program) # now unnecessary
            proportion = f"good/bad count = {100/count}%"
            return str_program, proportion
    
@timeitlogs
def write_random_ARC_walks(n_walks: int, len_walk: int, batch_write_size: int, program_length = None, n_grams: int = 1):
    assert program_length == None, "For now only length 4"

    ARC_dsl:ARC_DSL =pickle.load(open("DSL\\arc_dsl_8type.pkl", "rb"))
    if n_grams == 1:
        ARC_cfg:ARC_CFG=pickle.load(open("DSL\\arc_1gram_8type.pkl", "rb"))
    if n_grams == 2:
        ARC_cfg:ARC_CFG=pickle.load(open("DSL\\arc_2gram_8type.pkl", "rb"))
    ARC_unif_pcfg: ARC_PCFG = ARC_cfg.CFG_to_Uniform_PCFG()
    program_generator = ARC_unif_pcfg.sampling()

    
    # load ARC matrices sampling function
    # from sample_random_ARC_grid import sample_random_ARC_grid_full as ARC_grid_sampler
    from sample_random_ARC_grid import sample_random_ARC_grid_partial as ARC_grid_sampler

    fo = open("DSL\\ARC_random_walks.txt","w")
    isNotLast = True
    for batch_index in tqdm(range(n_walks//(batch_write_size*2))):
        to_write: list[str] = []
        # each for loop writes two sequences: one from one ARC inp. grid, the other from the corr. arc out. grid
        for walk_index in range(batch_write_size):
            if batch_index==n_walks//(batch_write_size*2)-1 and walk_index==batch_write_size-1: isNotLast=False
            io = ARC_grid_sampler(40)
            inp = io["input"]
            out = io["output"]
            first_seq = get_one_random_ARC_walk(inp, len_walk, program_generator, ARC_dsl)
            second_seq = get_one_random_ARC_walk(out, len_walk, program_generator, ARC_dsl)
            to_write.append(first_seq + "\n")
            to_write.append(second_seq + "\n"*isNotLast)
        # fo.write(str(to_write))
        fo.writelines(to_write)
    fo.close()

from mask_ARC_programs import mask_program

@timeitlogs
def write_random_ARC_good_programs(n_programs: int, max_program_depth: int, batch_write_size: int, n_grams: int=1, fo_name = "DSL\\ARC_random_good_programs.txt", masked = False):
    assert max_program_depth == 4, "For now only length 4"

    ARC_dsl:ARC_DSL =pickle.load(open("DSL\\arc_dsl_8type.pkl", "rb"))
    if n_grams == 1:
        ARC_cfg:ARC_CFG=pickle.load(open("DSL\\arc_1gram_8type.pkl", "rb"))
    if n_grams == 2:
        ARC_cfg:ARC_CFG=pickle.load(open("DSL\\arc_2gram_8type.pkl", "rb"))
    ARC_unif_pcfg: ARC_PCFG = ARC_cfg.CFG_to_Uniform_PCFG()
    program_generator = ARC_unif_pcfg.sampling()
    
    from sample_random_ARC_grid import sample_random_ARC_grid_partial as ARC_grid_sampler

    fo = open(fo_name,"w")
    num_batches = n_programs//(batch_write_size*2)
    isNotLast=True
    for batch_index in tqdm(range(num_batches)): # 2* because each iteration writes 2 programs
        to_write: list[str] = []
        # each for loop writes two sequences: one from one ARC inp. grid, the other from the corr. arc out. grid
        for pr_index in range(batch_write_size):
            if batch_index==num_batches-1 and pr_index==batch_write_size-1: isNotLast=False
            io = ARC_grid_sampler(40)
            inp = io["input"]
            out = io["output"]
            first_program, first_ratio = get_one_random_ARC_program(inp, program_generator, ARC_dsl)
            second_program, second_ratio = get_one_random_ARC_program(out, program_generator, ARC_dsl)
            if not masked:
                to_write.append(first_program+"\n")
                to_write.append(second_program+"\n"*isNotLast)
            else:
                to_write.extend(str(couple)+"\n" for couple in mask_program(first_program, has_expansion=True))
                to_write.extend(str(couple)+"\n" for couple in mask_program(second_program, has_expansion=True))
                if not isNotLast: to_write[-1][:-1]
        fo.writelines(to_write)
    fo.close()

class ARCWalksDataset(IterableDataset): 
    def __init__(self, len_walk: int = 1, program_length = None, n_grams: int = 1, matrices_pool_size = 40, encode_tokens=True, masked = False, max_length = 50):
        super(ARCWalksDataset).__init__()
        assert program_length == None, "For now only length 4"

        self.ARC_dsl:ARC_DSL =pickle.load(open("DSL\\arc_dsl_8type.pkl", "rb"))
        if n_grams == 1:
            self.ARC_cfg:ARC_CFG=pickle.load(open("DSL\\arc_1gram_8type.pkl", "rb"))
        if n_grams == 2:
            self.ARC_cfg:ARC_CFG=pickle.load(open("DSL\\arc_2gram_8type.pkl", "rb"))
        self.ARC_unif_pcfg: ARC_PCFG = self.ARC_cfg.CFG_to_Uniform_PCFG()
        self.program_generator = self.ARC_unif_pcfg.sampling()

        self.len_walk = len_walk
        self.matrices_pool_size = matrices_pool_size

        from sample_random_ARC_grid import sample_random_ARC_grid_partial as ARC_grid_sampler
        self.ARC_grid_sampler = ARC_grid_sampler
        from manipulate_ARC_images import from_coupling_to_tensor_with_PIL
        self.from_coupling_to_tensor_with_PIL = from_coupling_to_tensor_with_PIL

        self.encode_tokens = encode_tokens
        self.masked = masked
        self.corpus = Corpus(path = None, masked=self.masked)
        self.max_length = max_length

    def __iter__(self):
        while True:
            # each for loop writes two sequences: one from one ARC inp. grid, the other from the corr. arc out. grid
            io = self.ARC_grid_sampler(self.matrices_pool_size)
            inp = io["input"]
            out = io["output"]
            inp1, pr_str1, pr_str_masked1, pr_1, out1 = get_one_random_ARC_walk(inp, self.len_walk, self.program_generator, self.ARC_dsl, masked=self.masked)
            inp2, pr_str2, pr_str_masked2, pr_2, out2 = get_one_random_ARC_walk(out, self.len_walk, self.program_generator, self.ARC_dsl, masked=self.masked)

            # inp1 = [[1,0,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]]
            #out1 = [[0,1,0,0],[0,1,1,0],[0,1,0,0]]
            # inp1=tuple(tuple(r) for r in inp1)
            #out1=tuple(tuple(r) for r in out1)

            img_tog_tensor1=self.from_coupling_to_tensor_with_PIL(inp1,out1) #3.224.224
            img_tog_tensor2=self.from_coupling_to_tensor_with_PIL(inp2,out2) #3.224.224

            if self.encode_tokens:
                pr_str1 = self.corpus.tokenize_input_sentence(pr_str1, max_length=self.max_length)
                pr_str2 = self.corpus.tokenize_input_sentence(pr_str2, max_length=self.max_length)
                pr_str_masked1 = self.corpus.tokenize_input_sentence(pr_str_masked1, max_length=self.max_length)
                pr_str_masked2 = self.corpus.tokenize_input_sentence(pr_str_masked2, max_length=self.max_length)

            if self.masked:
                yield {'im1':img_tog_tensor1,'pr_str1': pr_str1, 'pr_str_masked1': pr_str_masked1, #'pr1': pr_1, 
                       'im2':img_tog_tensor2,'pr_str2': pr_str2, 'pr_str_masked2': pr_str_masked2}#, #'pr2': pr_2}} # the programs cannot be put inside sdataloader 
            else:
                yield {'im1':img_tog_tensor1,'pr_str1': pr_str1, #'pr1': pr_1, 
                       'im2':img_tog_tensor2,'pr_str2': pr_str2}#, #'pr2': pr_2}} # the programs cannot be put inside sdataloader 



if __name__ == "__main__":
    # st = perf_counter()
    # size of file generated ca n_walks*len_walk KB ==> 3Mib in 100s ==> 3Gib <2000m=33h
    ds = ARCWalksDataset(masked=True)
    # for x in tqdm(range(100)):
    d = next(iter(ds))
    d = next(iter(ds))
    d = next(iter(ds))
    print(type(d["im1"]))
    print(type(d["pr_str1"]))
    print(d["pr_str1"])
    c:Corpus = ds.corpus
    print(c.dictionary.word2idx)
    # print(type(im))
    # print(im.shape)
    # arr=im.permute(1,2,0).numpy()
    # print(arr.shape)
    # import numpy as np
    # from_RGB_tensor_to_pil_image(im).show()
    # from_RGB_tensor_to_pil_image(y[0]).show()
    # print(pr)

    # from torch.utils.data import DataLoader
    # dl = DataLoader(ds, batch_size=3)
    # z=next(iter(dl))
    # print(z)
    # print(z['im1'].shape)


    # write_random_ARC_walks(n_walks=100, len_walk=1, batch_write_size=50) # ca O(n_walks^1)*O(len_walk^1)*O(batch_write_size^0)
    # write_random_ARC_good_programs(n_programs=100,batch_write_size=1,max_program_depth=4, n_grams=1)
    # write_random_ARC_good_programs(n_programs=3000,batch_write_size=100,max_program_depth=4, n_grams=1, masked = True, fo_name="DSL\\ARC_random_good_masked_programs_test.txt")
    # write_random_ARC_good_programs(n_programs=3000,batch_write_size=100,max_program_depth=4, n_grams=1, masked = True, fo_name="DSL\\ARC_random_good_masked_programs_val.txt")
    # write_random_ARC_good_programs(n_programs=100,batch_write_size=1,max_program_depth=4, n_grams=1)
    # write_random_ARC_good_programs(n_programs=1000,batch_write_size=1,max_program_depth=4, n_grams=1)
    # write_random_ARC_good_programs(n_programs=10000,batch_write_size=1,max_program_depth=4, n_grams=1)
    # en = perf_counter()
    # fo = open("performances.txt", "a")
    # fo.write(f"function: write_random_ARC_walks(100, 10, 10) time: {en-st}")
    # print(en-st)