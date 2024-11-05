import os
os.environ["PYTHONHASHSEED"] = "seed"

import DSL.circuits as circuits
import DSL.list as dreamcoder
from type_system import *

import dsl

def stop():
    raise TypeError("stop")


# from produce_network import * 

# stop()

# or_fun = circuits.semantics["or"]
# for x in ["True", "False"]:
#     or_fun = or_fun(x)
# print(or_fun)

# stop()
class Print():
    def __lt__(self,x):
        print(x)
p = Print()
# p<1

#p<DSL.circuits.primitive_types['not']
#p<DSL.circuits.semantics['not']


# neg = circuits.semantics['not']
# o = circuits.semantics['or']
# print(o(True)(False), neg(True))



#p<cfg.CFG.__doc__
#p<dsl.DSL.__doc__

# dreamcoder_dsl = dsl.DSL(primitive_types=dreamcoder.primitive_types, semantics=dreamcoder.semantics)#, no_repetitions=dreamcoder.no_repetitions)
# dreamcoder_cfg = dreamcoder_dsl.DSL_to_CFG(type_request = Arrow(List(INT), List(INT)))
# dreamcoder_pcfg = dreamcoder_cfg.CFG_to_Uniform_PCFG()
# from pprint import pprint
#pprint(vars(dreamcoder_pcfg))
# fo=open("del.txt","w")
# fo.write(str(dreamcoder_pcfg))
# fo.close()
# print(dreamcoder_pcfg)
# stop()
# circuits_dsl = dsl.DSL(primitive_types=circuits.primitive_types, semantics=circuits.semantics, no_repetitions=circuits.no_repetitions)
# circuits_cfg = circuits_dsl.DSL_to_CFG(type_request = Arrow(BOOL, Arrow(BOOL, BOOL)),n_gram=1)
# p<circuits_cfg
# p<circuits_cfg.CFG_to_Random_PCFG()
# circuits_cfg = circuits_dsl.DSL_to_CFG(type_request = Arrow(BOOL, Arrow(BOOL, BOOL)),n_gram=2)
# p<circuits_cfg
# p<circuits_cfg.CFG_to_Random_PCFG()
# stop()

# import json
# import pickle
# d = {"start": circuits_cfg.start,
# "rules": circuits_cfg.rules,
# "max_program_depth": circuits_cfg.max_program_depth,}
# #with open("del_cfg.py","w") as fo:
# # json.dump(d, open("del_cfg.py","w"), indent=4, sort_keys=False)
# pickle.dump(circuits_cfg, open("del_cfg.py","wb"))
# dd = pickle.load(open("del_cfg.py","rb"))
# print(dd)
# print(dd.max_program_depth)
# print(dd == circuits_cfg)
# stop()
# circuits_cfg = circuits_dsl.DSL_to_CFG(type_request = Arrow(BOOL, BOOL), n_gram=2, max_program_depth=3)
# print(circuits_dsl)
# p<circuits_dsl.primitive_types()
# p<circuits_dsl.return_types()
# p<circuits_dsl.all_type_requests(0)
# p<circuits_dsl.all_type_requests(1)
# p<circuits_dsl.all_type_requests(2)
# p<circuits_dsl.all_type_requests(3)
# p<circuits_dsl.all_type_requests(4)
#print(circuits_cfg);stop()
# circuits3_cfg = circuits_dsl.DSL_to_CFG(type_request = Arrow(BOOL, Arrow(BOOL, Arrow(BOOL, BOOL))))
# circuits4_cfg = circuits_dsl.DSL_to_CFG(type_request = Arrow(BOOL, Arrow(BOOL, Arrow(BOOL, Arrow(BOOL, BOOL)))))
# type4 = Arrow(BOOL, Arrow(BOOL, Arrow(BOOL, Arrow(BOOL, BOOL))))



# circuits_pcfg = circuits_cfg.CFG_to_Uniform_PCFG()
# circuits3_pcfg = circuits4_cfg.CFG_to_Uniform_PCFG()
# circuits4_pcfg = circuits4_cfg.CFG_to_Uniform_PCFG()

pr = None
# for i,program in enumerate(circuits_pcfg.sampling()):
#     print(program)
#     print(type(program))
#     print(vars(program))
#     pr = program
#     if i > 10:
#         break

# print(pr)
# print("EVAL>>>>", pr.eval_naive(dsl = circuits_dsl, environment=[True, False]))
# print("EVAL>>>>", pr.eval_naive(dsl = circuits_dsl, environment=[False, True]))
# print("EVAL>>>>", pr.eval_naive(dsl = circuits_dsl, environment=[False, False]))
# print("EVAL>>>>", pr.eval_naive(dsl = circuits_dsl, environment=[True, True]))



import experiment_helper
implication_examples = [
    ([True,True], True),
    ([False,True], True),
    ([False,False], True),
    ([True,False], False),
]
implication_AandB = [
    ([True,True], True),
    ([False,True], False),
    ([False,False], False),
    ([True,False], False),
]
implication_AentailsB = [
    ([True,True], True),
    ([False,True], True),
    ([False,False], True),
    ([True,False], False),
]
implication_andABCD = [
    ([True,True,True,True], True),
    ([True,True,True,False], False),
    ([True,True,False,True], False),
    ([True,True,False,False], False),
    ([True,False,True,True], False),
    ([True,False,True,False], False),
    ([True,False,False,True], False),
    ([True,False,False,False], False),
    ([False,True,True,True], False),
    ([False,True,True,False], False),
    ([False,True,False,True], False),
    ([False,True,False,False], False),
    ([False,False,True,True], False),
    ([False,False,True,False], False),
    ([False,False,False,True], False),
    ([False,False,False,False], False),
]
implication_falsityABCD = [
    ([True,True,True,True], False),
    ([True,True,True,False], False),
    ([True,True,False,True], False),
    ([True,True,False,False], False),
    ([True,False,True,True], False),
    ([True,False,True,False], False),
    ([True,False,False,True], False),
    ([True,False,False,False], False),
    ([False,True,True,True], False),
    ([False,True,True,False], False),
    ([False,True,False,True], False),
    ([False,True,False,False], False),
    ([False,False,True,True], False),
    ([False,False,True,False], False),
    ([False,False,False,True], False),
    ([False,False,False,False], False),
]
test = [
    ([True,True], False),
    #([False,True], False),
    #([False,False], False),
    #([True,False], False),
]
# circuits_checks_AandB = experiment_helper.make_program_checker(circuits_dsl, implication_AandB) #-> Callable[[Program, bool], bool]
# circuits_checks_AentailsB = experiment_helper.make_program_checker(circuits_dsl, implication_AentailsB) #-> Callable[[Program, bool], bool]
# circuits4_checks_andABCD = experiment_helper.make_program_checker(circuits_dsl, implication_andABCD) #-> Callable[[Program, bool], bool]
# circuits4_checks_falsityABCD = experiment_helper.make_program_checker(circuits_dsl, implication_falsityABCD) #-> Callable[[Program, bool], bool]
# p<circuits_checks_AandB(program, False)
# p<circuits_checks_AentailsB(program, False)
# p<program

import run_experiment
# grammar with wrong type will never find solution
#out = run_experiment.run_algorithm(is_correct_program = circuits4_checks_andABCD, pcfg = circuits3_pcfg, algo_index = 0)
# grammar with correct type (maybe) will find solution
# out = run_experiment.run_algorithm(is_correct_program = circuits4_checks_andABCD, pcfg = circuits4_pcfg, algo_index = 0)
# program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
# print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
# print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
algo_map = '''
0 => Heap Search
1 => SQRT
2 => Threshold
3 => Sort & Add
4 => DFS
5 => BFS
6 => A*
'''

import model_loader

type4 = Arrow(BOOL, Arrow(BOOL, Arrow(BOOL, Arrow(BOOL, BOOL))))
dsl4,cfg_dict4,model4 = model_loader.build_circuits_generic_model(types={type4}
                                                  )
import Predictions.dataset_sampler
import Predictions.IOencodings
ds4 = Predictions.dataset_sampler.Dataset(
    size=100,
    dsl=dsl4,
    pcfg_dict={type4: cfg_dict4[type4].CFG_to_Uniform_PCFG()},
    nb_examples_max=5,
    arguments={type4: type4.arguments()}, # {[bool, bool, bool, bool]}
    # IOEncoder = IOEncoder,
    # IOEmbedder = IOEmbedder,
    ProgramEncoder=model4.ProgramEncoder,
    size_max=model4.IOEncoder.size_max,
    #This just exstracts the original lexicon (minus paddings, and start/stop tokens)
    lexicon=model4.IOEncoder.lexicon[:-2] if isinstance(
        model4.IOEncoder, Predictions.IOencodings.FixedSizeEncoding) else model4.IOEncoder.lexicon[:-4],
    for_flashfill=False
)

#next(iter(ds4))
