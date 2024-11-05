import dill as pickle
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

ARC_dsl:ARC_DSL =pickle.load(open("DSL\\arc_dsl_8type.pkl", "rb"))
ARC_cfg_1:ARC_CFG=pickle.load(open("DSL\\arc_1gram_8type.pkl", "rb"))

ARC_unif_pcfg: ARC_PCFG = ARC_cfg_1.CFG_to_Uniform_PCFG()
program_generator = ARC_unif_pcfg.sampling()


def get_one_random_ARC_walk(ARC_grid: list[list[int]], len_walk: int) -> str:#list[tuple[list[list[int]],str]]:
    """Will immediately reformat ARC_grid input to Tuple[Tuple[int]]"""
    good = 0
    count = 0
    #cur_list = [(ARC_grid, "START")]
    ARC_grid = tuple(tuple(row) for row in ARC_grid)
    cur_list = ["["+str(ARC_grid).replace(" ","")+",'*START*']"]
    out = ARC_grid
    while len(cur_list) < len_walk:
        program = next(program_generator)
        count += 1
        inp = out
        out = program.eval_naive(dsl = ARC_dsl, environment=[inp])
        if out != None and all(out) and len(out) < 30 and len(out) > 1:
            if isinstance(out[0],int):
                raise TypeError(f"Int found from {program} with out = {out} and input = {inp}")
            if len(frow:=out[0]) > 1 and (el:=frow[0]) != None and el > 0:
                str_program = str(program)
                if "var0" in str_program:
                    good +=1
                    # cur_list.append((out,str_program))
                    cur_list.append("["+str(out).replace(" ","")+",'"+str_program+"']")
    print(f"good, count, good/count: {good}, {count}, {good/count}")
    # return cur_list
    return "["+ ",".join(cur_list) +"]\n"

@timeitlogs
def write_random_ARC_walks(n_walks: int, len_walk: int, batch_write_size: int, program_length = None):
    assert program_length == None, "For now only length 4"

    # load ARC matrices sampling function
    # from sample_random_ARC_grid import sample_random_ARC_grid_full as ARC_grid_sampler
    from sample_random_ARC_grid import sample_random_ARC_grid_partial as ARC_grid_sampler

    fo = open("DSL\\ARC_random_walks.txt","w")
    for batch_index in tqdm(range(n_walks//batch_write_size)):
        to_write: list[str] = []
        # each for loop writes two sequences: one from one ARC inp. grid, the other from the corr. arc out. grid
        for walk_index in range(batch_write_size):
            io = ARC_grid_sampler(40)
            inp = io["input"]
            out = io["output"]
            first_seq = get_one_random_ARC_walk(inp, len_walk)
            second_seq = get_one_random_ARC_walk(out, len_walk)
            to_write.append(first_seq)
            to_write.append(second_seq)
        # fo.write(str(to_write))
        fo.writelines(to_write)
    fo.close()

if __name__ == "__main__":
    # st = perf_counter()
    # size of file generated ca n_walks*len_walk KB ==> 3Mib in 100s ==> 3Gib <2000m=33h
    write_random_ARC_walks(n_walks=1000, len_walk=3+1, batch_write_size=50) # ca O(n_walks^1)*O(len_walk^1)*O(batch_write_size^0)
    # en = perf_counter()
    # fo = open("performances.txt", "a")
    # fo.write(f"function: write_random_ARC_walks(100, 10, 10) time: {en-st}")
    # print(en-st)