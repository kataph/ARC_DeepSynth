import torch 
import logging
import argparse
import matplotlib.pyplot as plt

import deepcoder_dataset_loader
from type_system import Arrow, List, INT
from Predictions.IOencodings import FixedSizeEncoding
from Predictions.models import RulesPredictor, BigramsPredictor
from Predictions.dataset_sampler import Dataset

logging_levels = {0:logging.INFO, 1:logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args,unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

from model_loader import build_deepcoder_generic_model, build_deepcoder_intlist_model, build_dreamcoder_intlist_model, build_flashfill_generic_model, get_model_name


## HYPERPARMETERS

dataset_name = "dreamcoder"
# dataset_name = "deepcoder"
# dataset_name = "flashfill"

# Set to None for model invariant of type request
type_request = Arrow(List(INT), List(INT))
# type_request = None

dataset_size: int = 10_000
nb_epochs: int = 1
batch_size: int = 128

## TRAINING

if dataset_name == "dreamcoder":
    cur_dsl, cfg, model = build_dreamcoder_intlist_model()
#dataset_name = "deepcoder"
elif dataset_name == "deepcoder":
    if type_request is None:
        _, type_requests = deepcoder_dataset_loader.load_tasks("./deepcoder_dataset/T=3_test.json")
        cur_dsl, cfg_dict, model = build_deepcoder_generic_model(type_requests)
    else:
        cur_dsl, cfg, model = build_deepcoder_intlist_model()
elif dataset_name == "flashfill":
    cur_dsl, cfg_dict, model = build_flashfill_generic_model()
else:
    assert False, f"Unrecognized dataset: {dataset_name}"


print("Training model:", get_model_name(model), "on", dataset_name)
print("Type Request:", type_request or "generic")

if type_request:
    nb_examples_max: int = 2
else:
    nb_examples_max: int = 5

############################
######## TRAINING ##########
############################

def train(model, dataset):
    savename = get_model_name(model) + "_" + dataset_name + ".weights"
    for epoch in range(nb_epochs):
        gen = dataset.__iter__()
        for i in range(dataset_size // batch_size):
            batch_IOs, batch_program, batch_requests = [], [], []
            for _ in range(batch_size):
                io, prog, _ , req= next(gen)
                # FC: debug
                # THIS HAD AN [0] TOO MUCH!!! query = _.eval_naive(dataset.dsl, io[0][0][0])
                # end debug
                first_io = io[0]
                input_list, out_list = first_io
                flag = 0
                if len(input_list) > 1:
                    flag = 1
                else:
                    flag = 0
                # end debug
                batch_IOs.append(io)
                batch_program.append(prog)
                batch_requests.append(req)
            model.optimizer.zero_grad()
            # print("batch_program", batch_program.size())
            batch_predictions = model(batch_IOs)
            # print("batch_predictions", batch_predictions.size())
            if isinstance(model, RulesPredictor):
                loss_value = model.loss(
                    batch_predictions, torch.stack(batch_program))
            elif isinstance(model, BigramsPredictor):
                batch_grammars = model.reconstruct_grammars(
                    batch_predictions, batch_requests)
                loss_value = model.loss(
                    batch_grammars, batch_program)
            loss_value.backward()
            model.optimizer.step()
            print("\tminibatch: {}\t loss: {} metrics: {}".format(i, float(loss_value), model.metrics(loss=float(loss_value), batch_size=batch_size)))

        print("epoch: {}\t loss: {}".format(epoch, float(loss_value)))
        torch.save(model.state_dict(), savename)

def print_embedding(model):
    print(model.IOEmbedder.embedding.weight)
    print([x for x in model.IOEmbedder.embedding.weight[:, 0]])
    x = [x.detach().numpy() for x in model.IOEmbedder.embedding.weight[:, 0]]
    y = [x.detach().numpy() for x in model.IOEmbedder.embedding.weight[:, 1]]
    label = [str(a) for a in model.IOEncoder.lexicon]
    plt.plot(x,y, 'o')
    for i, s in enumerate(label):
        xx = x[i]
        yy = y[i]
        plt.annotate(s, (xx, yy), textcoords="offset points", xytext=(0,10), ha='center')
    plt.show()

# def test():
#     (batch_IOs, batch_program) = next(dataloader)
#     batch_predictions = model(batch_IOs)
#     batch_grammars = model.reconstruct_grammars(batch_predictions)
#     for program, grammar in zip(batch_program, batch_grammars):
#         # print("predicted grammar {}".format(grammar))
#         print("intended program {}\nprobability {}".format(
#             program, grammar.probability_program(model.cfg.start, program)))

def build_dataset_and_train(cur_dsl, cfg, model):
    dataset = Dataset(
        size=dataset_size,
        dsl=cur_dsl,
        pcfg_dict={type_request: cfg.CFG_to_Uniform_PCFG()} if type_request else {
            t: cfg.CFG_to_Uniform_PCFG() for t, cfg in cfg_dict.items()},
        nb_examples_max=nb_examples_max,
        arguments={type_request: type_request.arguments()} if type_request else {
            t: t.arguments() for t in cfg_dict.keys()},
        # IOEncoder = IOEncoder,
        # IOEmbedder = IOEmbedder,
        ProgramEncoder=model.ProgramEncoder,
        size_max=model.IOEncoder.size_max,
        lexicon=model.IOEncoder.lexicon[:-2] if isinstance(
            model.IOEncoder, FixedSizeEncoding) else model.IOEncoder.lexicon[:-4],
        for_flashfill=dataset_name == "flashfill"
    )

    train(model, dataset)

build_dataset_and_train(cur_dsl, cfg, model)
# build_dataset_and_train(cur_dsl_bi, cfg_bi, model_bi)

# test()
###print_embedding(model)

import experiment_helper
task_name = "car_task"
task_examples = [
    #([list int],list int)
    ([[14,-1,34,4,1]], [14]),
    ([[11,2-29,5,2]], [11]),
    ([[10,-10,3,2,0]], [10]),
    ([[0,1,2]], [0]),
    ([[14]], [14]),
]
# FC: I verified that the solution found, (map car (append var0 empty)), is actually a solution!
# You evaluate it like this: map(car)(append(var0)(empty)), and if var0 = [14,-1,34,4,1] then output = [14]
# Beautiful!
tasks = [(task_name,task_examples)]
conditioned_grammars = experiment_helper.task_set2dataset(tasks = tasks, model = model, dsl = cur_dsl)
assert len(conditioned_grammars) == 1, "Should be just one task, hence one grammar"
name, pcfg_out, pr_checker_out = conditioned_grammars[0]
print(name)

fo = open("del2.txt","w")
fo.write(str(pcfg_out))
fo.close()
print("All done! Check file del2 and compare probs with del")

dreamcoder_uniform_pcfg = cfg.CFG_to_Uniform_PCFG()

import run_experiment
print("Results for the UNIFROM PCFG")
out = run_experiment.run_algorithm(is_correct_program = pr_checker_out, pcfg = dreamcoder_uniform_pcfg, algo_index = 0)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
print("Results for the UNIFROM PCFG 2")
out = run_experiment.run_algorithm(is_correct_program = pr_checker_out, pcfg = dreamcoder_uniform_pcfg, algo_index = 0)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)

print("Results for the MODEL-CONDITIONED PCFG")
out = run_experiment.run_algorithm(is_correct_program = pr_checker_out, pcfg = pcfg_out, algo_index = 0)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
print("Results for the MODEL-CONDITIONED PCFG 2")
out = run_experiment.run_algorithm(is_correct_program = pr_checker_out, pcfg = pcfg_out, algo_index = 0)
program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out
print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)


# conditioned_grammars_bi = experiment_helper.task_set2dataset(tasks = tasks, model = model_bi, dsl = cur_dsl_bi)
# assert len(conditioned_grammars_bi) == 1, "Should be just one task, hence one grammar"
# name_bi, pcfg_out_bi, pr_checker_out_bi = conditioned_grammars_bi[0]
# assert name == name_bi

# fo = open("del3.txt","w")
# fo.write(str(pcfg_out_bi))
# fo.close()
# print("Results for the BIGRAM MODEL-CONDITIONED PCFG")
# out_bi = run_experiment.run_algorithm(is_correct_program = pr_checker_out_bi, pcfg = pcfg_out_bi, algo_index = 0)
# program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out_bi
# print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
# print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
# print("Results for the BIGRAM MODEL-CONDITIONED P_programs, cumulative_probability, probability")

# print("Results for the BIGRAM MODEL-CONDITIONED PCFG2")
# out_bi = run_experiment.run_algorithm(is_correct_program = pr_checker_out_bi, pcfg = pcfg_out_bi, algo_index = 0)
# program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability = out_bi
# print("program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability")
# print(program_r, search_time, evaluation_time, nb_programs, cumulative_probability, probability)
# print("Results for the BIGRAM MODEL-CONDITIONED P_programs, cumulative_probability, probability")

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>END<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")