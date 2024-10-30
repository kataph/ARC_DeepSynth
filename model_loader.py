import os
import typing
import torch
from type_system import INT, STRING, Arrow, List, Type
from typing import Dict, Set, Tuple
from cfg import CFG
import dsl
from DSL import list, deepcoder, flashfill, circuits
from Predictions.IOencodings import FixedSizeEncoding
from Predictions.embeddings import RNNEmbedding, SimpleEmbedding
from Predictions.models import RulesPredictor, BigramsPredictor, NNDictRulesPredictor


def __block__(input_dim, output_dimension, activation):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dimension),
        activation,
    )

def get_model_name(model) -> str:
    name: str = ""
    if isinstance(model.IOEncoder, FixedSizeEncoding):
        name += "fixed"
    else:
        name += "variable"
    if isinstance(model.IOEmbedder, SimpleEmbedding):
        name += "+simple"
    else:
        name += "+rnn"
    if isinstance(model, NNDictRulesPredictor):
        name += "+nndict_rules"
    elif isinstance(model, RulesPredictor):
        name += "+rules"
    else:
        name += "+bigrams"
    return name


def __buildintlist_model(dsl: dsl.DSL, max_program_depth: int, nb_arguments_max: int, lexicon: typing.List[int], size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int) -> Tuple[CFG, RulesPredictor]:
    type_request = Arrow(List(INT), List(INT))
    cfg = dsl.DSL_to_CFG(
        type_request, max_program_depth=max_program_depth)

    ############################
    ###### IO ENCODING #########
    ############################

    # IO = [[I1, ...,Ik], O]
    # I1, ..., Ik, O are lists
    # IOs = [IO1, IO2, ..., IOn]
    # task = (IOs, program)
    # tasks = [task1, task2, ..., taskp]

    #### Specification: #####
    # IOEncoder.output_dimension: size of the encoding of one IO
    # IOEncoder.lexicon_size: size of the lexicon
    # IOEncoder.encode_IO: outputs a tensor of dimension IOEncoder.output_dimension
    # IOEncoder.encode_IOs: inputs a list of IO of size n
    # and outputs a tensor of dimension n * IOEncoder.output_dimension

    IOEncoder = FixedSizeEncoding(
        nb_arguments_max=nb_arguments_max,
        lexicon=lexicon,
        size_max=size_max,
    )

    ############################
    ######### EMBEDDING ########
    ############################

    IOEmbedder = RNNEmbedding(
        IOEncoder=IOEncoder,
        output_dimension=embedding_output_dimension,
        size_hidden=size_hidden,
        number_layers_RNN=number_layers_RNN,
    )

    #### Specification: #####
    # IOEmbedder.output_dimension: size of the output of the embedder
    # IOEmbedder.forward_IOs: inputs a list of IOs
    # and outputs the embedding of the encoding of the IOs
    # which is a tensor of dimension
    # (IOEmbedder.input_dimension, IOEmbedder.output_dimension)
    # IOEmbedder.forward: same but with a batch of IOs

    ############################
    ######### MODEL ############
    ############################

    latent_encoder = torch.nn.Sequential(
        __block__(IOEncoder.output_dimension * IOEmbedder.output_dimension, size_hidden, torch.nn.Sigmoid()),
        # block(size_hidden, size_hidden, nn.LeakyReLU()),
        __block__(size_hidden, size_hidden, torch.nn.Sigmoid()),
    )

    model = RulesPredictor(
        cfg=cfg,
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        latent_encoder=latent_encoder,
    )


    return cfg, model


def build_dreamcoder_intlist_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[dsl.DSL, CFG, RulesPredictor]:
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_arguments_max = 1  # maximum number of inputs in an IO
    lexicon = [x for x in range(-30, 30)]  # all elements of a list must be from lexicon

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64


    dreamcoder = dsl.DSL(list.semantics, list.primitive_types)

    dreamcoder_cfg, model = __buildintlist_model(
        dreamcoder, max_program_depth, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)
    
    if autoload:
        weights_file = get_model_name(model) + "_dreamcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return dreamcoder, dreamcoder_cfg, model


def build_deepcoder_intlist_model(max_program_depth: int = 4, autoload: bool = True) -> Tuple[dsl.DSL, CFG, RulesPredictor]:
    size_max = 10  # maximum number of elements in a list (input or output)
    nb_arguments_max = 1  # maximum number of inputs in an IO
    # all elements of a list must be from lexicon
    lexicon = [x for x in range(-256, 256)]

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    deepcoder_dsl = dsl.DSL(deepcoder.semantics, deepcoder.primitive_types, deepcoder.no_repetitions)

    deepcoder_cfg, model = __buildintlist_model(
        deepcoder_dsl, max_program_depth, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_deepcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return deepcoder_dsl, deepcoder_cfg, model


def __build_generic_model(dsl: dsl.DSL, cfg_dictionary: Dict[Type, CFG], nb_arguments_max: int, lexicon: typing.List[int], size_max: int, size_hidden: int, embedding_output_dimension: int, number_layers_RNN: int) -> BigramsPredictor:
    IOEncoder = FixedSizeEncoding(
        nb_arguments_max=nb_arguments_max,
        lexicon=lexicon,
        size_max=size_max,
    )
    IOEmbedder = RNNEmbedding(
        IOEncoder=IOEncoder,
        output_dimension=embedding_output_dimension,
        size_hidden=size_hidden,
        number_layers_RNN=number_layers_RNN,
    )

    latent_encoder = torch.nn.Sequential(
        __block__(IOEncoder.output_dimension *
                  IOEmbedder.output_dimension, IOEncoder.output_dimension *
                  IOEmbedder.output_dimension // 2, torch.nn.ReLU()),
    )

    return BigramsPredictor(
        cfg_dictionary=cfg_dictionary,
        primitive_types={x: x.type for x in dsl.list_primitives},
        IOEncoder=IOEncoder,
        IOEmbedder=IOEmbedder,
        latent_encoder=latent_encoder
    )

# TODO: FC: the return types are wrong: CFG is not returned, but dict[Type,CFG] ({type:cfg, ...})
def build_deepcoder_generic_model(types: Set[Type], max_program_depth: int = 4, autoload: bool = True) -> Tuple[dsl.DSL, CFG, BigramsPredictor]:
    size_max = 19  # maximum number of elements in a list (input or output)
    nb_arguments_max = 3
    # all elements of a list must be from lexicon
    lexicon = [x for x in range(-256, 257)]

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    deepcoder_dsl = dsl.DSL(deepcoder.semantics, deepcoder.primitive_types, deepcoder.no_repetitions)

    deepcoder_dsl.instantiate_polymorphic_types()
    cfg_dict = {}
    for type_req in types:
        cfg_dict[type_req] = deepcoder_dsl.DSL_to_CFG(type_req,
                 max_program_depth=max_program_depth)
    print("Requests:", "\n\t" + "\n\t".join(map(str, cfg_dict.keys())))

    model = __build_generic_model(
        deepcoder_dsl, cfg_dict, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_deepcoder.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return deepcoder_dsl, cfg_dict, model


def build_flashfill_generic_model(max_program_depth: int = 7, autoload: bool = True) -> Tuple[dsl.DSL, CFG, BigramsPredictor]:
    # Import is done here because it needs additional dependencies
    from flashfill_dataset_loader import get_lexicon
    size_max = 30  # maximum number of elements in a list/string (input or output)
    nb_arguments_max = 2
    # all elements of a list must be from lexicon
    lexicon = get_lexicon()

    embedding_output_dimension = 10
    # only useful for RNNEmbedding
    number_layers_RNN = 1
    size_hidden = 64
    flashfill_dsl = dsl.DSL(flashfill.semantics,
                        flashfill.primitive_types, flashfill.no_repetitions)

    flashfill_dsl.instantiate_polymorphic_types()
    requests = flashfill_dsl.all_type_requests(nb_arguments_max)
    cfg_dict = {}
    for type_req in requests:
        # Skip if it contains a list list
        if any(ground_type.size() >= 3 for ground_type in type_req.list_ground_types()):
            continue
        if any(arg != STRING for arg in type_req.arguments()):
            continue
        # Why the try?
        # Because for request type: int -> list(list(int)) in a dsl.DSL without a method to go from int -> list(int)
        # Then there is simply no way to produce the correct output type
        # Thus when we clean the PCFG by removing useless rules, we remove the start symbol thus creating an error
        try:
            cfg_dict[type_req] = flashfill_dsl.DSL_to_CFG(
                type_req, max_program_depth=max_program_depth)
        except:
            continue
    print("Requests:", cfg_dict.keys())

    model = __build_generic_model(
        flashfill_dsl, cfg_dict, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_flashfill.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return flashfill_dsl, cfg_dict, model


###---FC
def build_circuits_generic_model(
        types: Set[Type], 
        max_program_depth: int =4, 
        autoload: bool = True,
        size_max = 12,  
        embedding_output_dimension = 10,
        number_layers_RNN = 1,
        size_hidden = 64,
        ) -> Tuple[dsl.DSL, dict[Type,CFG], BigramsPredictor]:
    '''
    FC-Attempt to build model for boolean circuits.

    inputs
        - types: the target program types  
        - max_program_depth: depth of the program (in the sense of the
            depth of the corresponding Abstract Syntax Tree?)
        - autoload: if true loads weights from file {mode.name}_circuits.weights,
        - size_max: maximum number of boolean variables in an input truth assignment 
            (output is always just one boolean). Note that this means that the encoder
            will take strings and encode them into a 2*size_max space, each lexicon
            element will correspond to an index, occupying size_max places; alternatingly
            with these indices there are indices corresponding to the symbols ["PAD", "NOTPAD"],
            which are always added to the lexicon in any FixedSizeEncoding. 

            --> NOTPAD is never called? Is it an error?
            --> After correction, the encoding of, say, [a,b,c,d] for size_max=8 is 
            [idx(a), idx("NOTPAD"), idx(b), idx("NOTPAD"), idx(c), idx("NOTPAD"), idx(d), idx("NOTPAD"), idx("PAD"), idx("PAD")]
            e.g. [True, False, True, False] becomes [0,3,1,3,0,3,1,3,2,2] (before was [0,2,1,2,0,2,1,2,2,2])
        
            --> Note that the encoding of a SINGLE INPUT works as described above, but the encoding of an 
            INPUT[A LIST OF INPUTS]-OUTPUT example [[i1,...,in],o] produces
            the concatenation enc(i1)*enc(i2)*...*enc(in)*enc(o), e.g. [[[True,0], [False,1], [True,0], [False,1]], [1,0]] becomes
            tensor([0, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    1, 3, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    0, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    1, 3, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    0, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
            --> Any error raised during encoding of the inputs is traduced into a 'row' of PADs 
            (for output the error is escalated)
        - embedding_output_dimension = 10: ???????
        
        # only useful for RNNEmbedding
        - number_layers_RNN = 1 
        - size_hidden = 64,
    '''
    # all elements in a truth table must be booleans
    lexicon = [True, False]
    
    # NOOOOO! only unary or binary functions are allowed (redundant with the dsl not containing multiple-input boolean gates)
    # It is the max length of the inputs 
    # NOOOOO! it is the max number of input examples in an input/output thing, max length of input is size_max
    nb_arguments_max = 4#2

    circuits_dsl = dsl.DSL(circuits.semantics, circuits.primitive_types, circuits.no_repetitions)

    #circuits_dsl.instantiate_polymorphic_types() no polymorphic types?
    cfg_dict = {}
    for type_req in types: # will always be an arrow of certain bool arrows
        cfg_dict[type_req] = circuits_dsl.DSL_to_CFG(type_req,
                                    max_program_depth=max_program_depth)
    print("Requests:", "\n\t" + "\n\t".join(map(str, cfg_dict.keys())))

    model = __build_generic_model(
        circuits_dsl, cfg_dict, nb_arguments_max, lexicon, size_max, size_hidden, embedding_output_dimension, number_layers_RNN)

    if autoload:
        weights_file = get_model_name(model) + "_circuits.weights"
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file))
            print("Loaded weights.")

    return circuits_dsl, cfg_dict, model