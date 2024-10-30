from pcfg import PCFG
import random
import logging 
import torch
from type_system import STRING, Type, List, INT, BOOL
from cons_list import tuple2constlist

from flashfill_dataset_loader import randomWord

class Dataset(torch.utils.data.IterableDataset):
    """
        Dataset as an iterator: gives a stream of tasks
        a task is (IOs, program)

        size: size of the dataset

        nb_examples_max: number of IOs in a task
        arguments: list of arguments for the program

        size_max: maximum number of elements in a list
        lexicon: possible values in a list
    """
    def __init__(self, 
        size, 
        dsl, 
        pcfg_dict, 
        nb_examples_max, 
        arguments,
        # IOEncoder,
        # IOEmbedder,
        ProgramEncoder,
        size_max,
        lexicon,
        for_flashfill=False
        ):
        super(Dataset).__init__()
        self.size = size
        self.dsl = dsl
        self.input_sampler = Input_sampler(size_max = size_max, lexicon = lexicon)
        self.program_sampler = {t: pcfg.sampling() for t, pcfg in pcfg_dict.items()}
        self.arguments = arguments
        self.allowed_types = list(self.program_sampler.keys())

            
        self.nb_examples_max = nb_examples_max
        # self.IOEncoder = IOEncoder
        # self.IOEmbedder = IOEmbedder
        self.ProgramEncoder = ProgramEncoder
        self.lexicon = lexicon
        self.for_flashfill = for_flashfill

    def __iter__(self):
        return (self.__single_data__() for i in range(self.size))

    def __single_data__(self):
        # print("generating...")
        flag = True
        output = None
        while flag or output == None:
            # First select a type
            selected_type: Type = random.choice(self.allowed_types)
            rtype = selected_type.returns()
            # print("Selected type:", selected_type)
            program = next(self.program_sampler[selected_type])
            while program.is_constant():
                program = next(self.program_sampler[selected_type])
         
            if self.for_flashfill:
                # We have to init the constants
                n = program.count_constants()
                constants = [randomWord() for _ in range(n)]
                program = program.derive_with_constants(constants)


            nb_IOs = random.randint(1, self.nb_examples_max)
            inputs = [[self.input_sampler.sample(type_) for type_ in self.arguments[selected_type]] for _ in range(nb_IOs)]

            try:
                outputs = []
                for input_ in inputs:
                    environment = tuple2constlist(input_)
                    output = program.eval_naive(self.dsl, environment)
                    if self.__output_validation__(output, rtype):
                        outputs.append(output)
                    else:
                        raise ValueError()
                flag = False
            except:
                pass
        # print("Inputs:", inputs)
        # print("Outputs:", outputs)
        # print("\tprogram:", program)
        IOs = [[I,O] for I,O in zip(inputs, outputs)]
        logging.debug('Found a program:\n{}\nand inputs:\n{}'.format(program,IOs))
        return IOs, self.ProgramEncoder(program), program, selected_type
        # return self.IOEncoder.encode_IOs(IOs), self.ProgramEncoder(program)
    

    # TODO: FC: this function will raise an exception if not using
    # only INT, STRING, or LIST thereof. For instance, if output = False,
    # then len(output) will raise and exception. 
    # in any case, the function checks that outputs are in the lexicon, that
    # lists are not too long or empty. 
    # --> Notice that this error will cause an unmanaged infinite loop, since
    #     to end the while loop the program output must be correctly checked at least once
    # --> Modified to account also for BOOL 
    def __output_validation__(self, output, return_type: Type):
        if not hasattr(output, '__iter__'):
            if output not in self.lexicon:
                return False
        if hasattr(output, '__iter__'):
            for e in output:
                if e not in self.lexicon:
                    return False
        if isinstance(output, int) and return_type == INT:
            return True
        if isinstance(output, bool) and return_type in [BOOL,bool]: #FC: i am not sure of which one to check
            return True
        if len(output) == 0 or len(output) > self.input_sampler.size_max: 
            return False
        return True

class Input_sampler():
    """
    Object to sample element of a given type together with constraints parameters 
    For now we can only sample from the base types string, int and bool. And any list types that is based on lists or base types.
    However we sample elements from a lexicon so we cannot sample both a int list and a string for example sincei nthat case the lexicon woul need to conatin integers and strings.

    size_max: max number of elements in an input 
    lexicon: admissible elements in a list
    """
    def __init__(self, size_max, lexicon) -> None:
        self.size_max = size_max
        self.lexicon = lexicon

    def sample(self, type: Type):
        if isinstance(type, List):
            size = random.randint(1, self.size_max)
            res = [self.sample(type.type_elt) for _ in range(size)]
            return res
        if type.__eq__(INT):
            return random.choice(self.lexicon)
        elif type.__eq__(STRING):
            size = random.randint(1, self.size_max)
            res = "".join([random.choice(self.lexicon) for _ in range(size)])
            return res
        elif type.__eq__(BOOL): #FC: note that this is indipendent from self.lexicon
            return random.random() > .5
        assert(False)

