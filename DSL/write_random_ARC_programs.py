import dill as pickle
from ARC_type_system import Arrow, GRID, FrozenSet

from ARC_dsl_cfg import ARC_DSL
from ARC_cfg_pcfg import ARC_CFG

from program import format_program_full
from ARC_formatted_dsl import primitive_types

ARC_dsl:ARC_DSL =pickle.load(open("DSL\\arc_dsl_8type.pkl", "rb"))
ARC_cfg_1:ARC_CFG=pickle.load(open("DSL\\arc_1gram_8type.pkl", "rb"))

ARC_unif_pcfg = ARC_cfg_1.CFG_to_Uniform_PCFG()



grid = ((1,1,2,3),(2,4,5,6))
x = "(cellwise (identity (dedupe_ti var0)) (pair (last_ct var0) (interval EIGHT NEG_ONE SEVEN)) SEVEN)"
# [(downscale (hupscale var0 NINE) FIVE), (last_cf (initset var0))]
# EVAL>>>> ((1, 1, 1, 1, 2, 2, 3, 3), (1, 1, 2, 3), (2, 4, 5, 6))
# (compress (lefthalf var0)), (first_ct (hsplit var0 ZERO)), (mostcommon_ct (repeat EIGHT TEN))
# errore h, w = len(grid), len(grid[0]) // n   by zero
good = 0
count = 0
# 35% good, very good
for i,program in enumerate(ARC_unif_pcfg.sampling()):
    count += 1
    print(program)
    print(program.arguments)
    # print(vars(program))
    pr = program
    # print(format_program_full(pr, types=primitive_types))
    # if count == 5: raise TypeError("1")
    try:
        out = pr.eval_naive(dsl = ARC_dsl, environment=[grid])
    except:
        out = None
    if out == None:
        print("None for program ", pr)
    if out != None:
        if all(out) and len(out) > 1 and len(out[0]) > 1:
            good +=1
            # if good > 4:
            #     break
            goodout = out
    if count == 1000:
        print(good, count, good/count)
        print(goodout)
        break
print("EVAL>>>>", pr.eval_naive(dsl = ARC_dsl, environment=[grid]))


croptest=[
    ([((1,2),(3,4))], ((1,),)),
    ([((4,2),(3,4))], ((4,),)),
    ([((6,2),(3,4))], ((6,),)),
    ([((2,2),(3,4))], ((2,),)),
    ([((5,2),(3,4))], ((5,),)),
    ([((5,2),(3,4))], ((5,),)),
]
add1test=[
    ([((1,2),(3,4))], ((1+1,3),(4,5))),
    ([((4,2),(3,4))], ((4+1,3),(4,5))),
    ([((6,2),(3,4))], ((6+1,3),(4,5))),
    ([((2,2),(3,4))], ((2+1,3),(4,5))),
    ([((5,2),(3,4))], ((5+1,3),(4,5))),
    ([((5,2),(3,4))], ((5+1,3),(4,5))),
]
rot90test=[
    ([((1,2),(3,4))], ((2,4),(1,3))),
    ([((4,2),(2,8))], ((2,8),(4,2))),
]
rot90bigtest=[
    ([((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))], ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2, 2, 2, 2, 2), (3, 3, 3, 3, 3, 3, 3, 3, 3, 3), (4, 4, 4, 4, 4, 4, 4, 4, 4, 4), (5, 5, 5, 5, 5, 5, 5, 5, 5, 5), (6, 6, 6, 6, 6, 6, 6, 6, 6, 6), (7, 7, 7, 7, 7, 7, 7, 7, 7, 7), (8, 8, 8, 8, 8, 8, 8, 8, 8, 8), (9, 9, 9, 9, 9, 9, 9, 9, 9, 9))),
]

import experiment_helper
# ARC_checks_croptest = experiment_helper.make_program_checker(ARC_dsl, croptest) #-> Callable[[Program, bool], bool]
# ARC_checks_add1test = experiment_helper.make_program_checker(ARC_dsl, add1test) #-> Callable[[Program, bool], bool]
# ARC_checks_rot90test = experiment_helper.make_program_checker(ARC_dsl, rot90test) #-> Callable[[Program, bool], bool]
test_cur = rot90bigtest
ARC_checks_cur = experiment_helper.make_program_checker(ARC_dsl, test_cur)
#print(circuits_checks_AandB(program, False)
# p<circuits_checks_AentailsB(program, False)
# p<program

import run_experiment
# grammar with wrong type will never find solution
#out = run_experiment.run_algorithm(is_correct_program = circuits4_checks_andABCD, pcfg = circuits3_pcfg, algo_index = 0)
# grammar with correct type (maybe) will find solution
# out = run_experiment.run_algorithm(is_correct_program = ARC_checks_rot90test, pcfg = ARC_unif_pcfg, algo_index = 0)
algo_map = '''
0 => Heap Search
1 => SQRT
2 => Threshold
3 => Sort & Add
4 => DFS
5 => BFS
6 => A*
'''
out = run_experiment.run_algorithm(is_correct_program = ARC_checks_cur, pcfg = ARC_unif_pcfg, algo_index = 0)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
out = run_experiment.run_algorithm(is_correct_program = ARC_checks_cur, pcfg = ARC_unif_pcfg, algo_index = 1)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
out = run_experiment.run_algorithm(is_correct_program = ARC_checks_cur, pcfg = ARC_unif_pcfg, algo_index = 6)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
out = run_experiment.run_algorithm(is_correct_program = ARC_checks_cur, pcfg = ARC_unif_pcfg, algo_index = 5)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)