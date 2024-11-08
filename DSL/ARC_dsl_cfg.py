from collections import deque
import copy
from tqdm import tqdm as loading_bar
#from functools import lru_cache # Ah, l'avessi saputo prima che c'era...

from ARC_type_system import * #Type, PolymorphicType, PolymorphicTypeOrPrimitiveArrow, ExceedinglyPolymorphicType, PrimitiveType, Arrow, Tuple, FrozenSet, Union, Couple, INT, BOOL
import sys
sys.path.append("C:\\Users\\Francesco\\Desktop\\github_repos\\ARC\\code\\auxillary_github_repos\\DeepSynth")
from ARC_program import Program, Function, Variable, BasicPrimitive, New
from ARC_cfg_pcfg import ARC_CFG

from itertools import combinations_with_replacement, product

class ARC_DSL:
    """
    Object that represents a domain specific language

    list_primitives: a list of primitives, either BasicPrimitive or New

    semantics: a dictionary of the form {P : f}
    mapping a program P to its semantics f
    for P a BasicPrimitive
    """

    def __init__(self, semantics, primitive_types, no_repetitions=None, upper_bound_type_size = 10):
        self.list_primitives = []
        self.semantics = {}
        self.no_repetitions = no_repetitions or set()
        self.upper_bound_type_size = upper_bound_type_size
        self.hashed_type_expansions = {}

        for p in primitive_types:
            formatted_p = str(p)
            if formatted_p in semantics:
                self.semantics[formatted_p] = semantics[formatted_p]
                P = BasicPrimitive(
                    primitive=formatted_p, type_=primitive_types[p], probability={}
                )
                self.list_primitives.append(P)
            else:
                assert False, "No New programs here!"
                P = New(body=p.body, type_=primitive_types[p], probability={})
                self.list_primitives.append(P)
    
    def primitive_types(self):
        set_basic_types = set()
        for P in self.list_primitives:
            set_basic_types_P = P.type.list_ground_types(polymorphic=False)
            for x in set_basic_types_P:
                set_basic_types.add(x)
        return list(set_basic_types)

    def return_types(self):
        set_basic_types = set()
        for P in self.list_primitives:
            set_basic_types.add(P.type.returns())
        return list(set_basic_types)


    def all_type_requests(self, nb_arguments_max: int):
        types = self.primitive_types()
        all_return_types = self.return_types()
        requests = []
        for nb_arguments in range(1, nb_arguments_max + 1):
            signatures = combinations_with_replacement(types, nb_arguments)
            for signature in signatures:
                for return_type in all_return_types:
                    cur_type = return_type
                    for el in signature:
                        cur_type = Arrow(el, cur_type)
                    requests.append(cur_type)
        return requests

    def __str__(self):
        s = "Print a DSL\n"
        for P in self.list_primitives:
            s = s + "{}: {}\n".format(P, P.type)
        return s

    def instantiate_polymorphic_types(self):#, upper_bound_type_size=10): # shifted to object inizialization
        '''FC: from the original list of primitives, whenever a polym. type is found,
           the primitive is replaced with an instantiated versions. E.g. List t0 > List INT; List List INT;
           List List List INT; List BOOL; ...; List BOOL->INT; ... (note that these are the types, they
           are always attributed to the same primitive)
           --> Notice that they can be instantiate only to either 
           (i) primitive types
           (ii) arrows of primitive types
           (iii) lists and lists of lists of primitive types
           provided that the length of the ensuing type is less than the upper bound
           Other constructors are ignored'''
        set_basic_types = set()
        for P in self.list_primitives:
            set_basic_types_P, set_polymorphic_types_P = P.type.decompose_type()
            set_basic_types = set_basic_types | set_basic_types_P

        # if len(set_basic_types)>0: 
        #     print("basic types", set_basic_types)
        #     exit()

        # set_types = set(set_basic_types)
        # assert set_types != set_basic_types #gives me error
        
        # originally, added List(x) and List(List(x)) to set_types for each x in set basic_types
        # now I set it to {PIECE, CELL, BOOL, INTEGER_SET, INT, INDICES, OBJECTS, INTEGER_TUPLE, Tuple(INT), 
        # OBJECT, GRID, NUMERICAL, PATCH, Tuple(GRID), ELEMENT}
        # for type_ in set_basic_types:
        #     new_type = List(type_)
        #     set_types.add(new_type)
        #     new_type = List(new_type)
        #     set_types.add(new_type)
        set_basic_types = {PIECE, CELL, BOOL, INTEGER_SET, INT, INDICES, OBJECTS, INTEGER_TUPLE, Tuple(INT), OBJECT, GRID, 
                           #NUMERICAL,
                           PATCH, Tuple(GRID), ELEMENT, 
                           Couple(PATCH,PATCH), Couple(OBJECT,OBJECT), Couple(INDICES,INDICES)}
        # set_types is the set of types used to instantiate the polymorphic types
        set_types = set(set_basic_types)

        # Adds Arrow(x,y) to set_types for x,y in set basic_types
        # for type_ in set_basic_types:
        #     for type_2 in set_basic_types:
        #         new_type2 = Arrow(type_, type_2)
        #         set_types.add(new_type2)
        # TODO: perlomeno anche alcune coppie
        {set_types.add(Arrow(type_, type_2)) for type_, type_2 in product(set_basic_types, set_basic_types)}

        # print("set_types", set_types)
        # exit()

        # unused?
        new_primitive_types = {}

        for P in self.list_primitives[:]:
            assert isinstance(P, (New, BasicPrimitive))
            type_P = P.type
            set_basic_types_P, set_polymorphic_types_P = type_P.decompose_type()
            #print(P, P.type, set_basic_types_P, set_polymorphic_types_P)
            if set_polymorphic_types_P:
                set_instantiated_types = set()
                set_instantiated_types.add(type_P)
                for poly_type in set_polymorphic_types_P: # e.g. t1 in {t0,t1,t2}
                    new_set_instantiated_types = set() 
                    for type_ in loading_bar(set_types): # e.g. (GRID->GRID) in {240 basic types + lv1 Arrows}
                        # Here go polymorphic-type-level filters
                        # eg if polym. type must not be intantiated by arrows:
                        # if isinstance(poly_type, NOARROWPOLYTYPE) and 
                        # type_ == Arrow: continue
                        # if poly_type.name == "n0": 
                        #     print("Flag")
                        if (isinstance(poly_type, PolymorphicTypeNoArrow) and 
                            isinstance(type_,Arrow)):
                            # assert False, "For now it should not be activated"
                            continue
                        for instantiated_type in set_instantiated_types: # e.g. (INT->t1->t2)=(t0->t1->t2) with t0 replaced by INT in {set of all partially instantiated types}
                            unifier = {str(poly_type): type_} # e.g. {"t1": GRID} 
                            intermediate_type = copy.deepcopy(instantiated_type)
                            new_type = intermediate_type.apply_unifier(unifier) # obtains e.g. (INT->GRID->t2)
                            if new_type.size() <= self.upper_bound_type_size:
                                new_set_instantiated_types.add(new_type)
                    set_instantiated_types = new_set_instantiated_types
                for type_ in set_instantiated_types:
                    if isinstance(P, New):
                        instantiated_P = New(P.body, type_, probability={})
                    if isinstance(P, BasicPrimitive):
                        instantiated_P = BasicPrimitive(
                            P.primitive, type_, probability={}
                        )
                    self.list_primitives.append(instantiated_P)
                self.list_primitives.remove(P) # programs with old uninstantiated-polymorphic-variables-types are removed

    def ARC_DSL_to_ARC_CFG(
        self,
        type_request,
        # upper_bound_type_size=10, # shifted to object initialization
        max_program_depth=4,
        min_variable_depth=1,
        n_gram=2,
    ):
        """
        Constructs a CFG from a DSL imposing bounds on size of the types
        and on the maximum program depth
        """
        self.instantiate_polymorphic_types()

        if isinstance(type_request, Arrow):
            return_type = type_request.returns()
            args = type_request.arguments()
        else:
            return_type = type_request
            args = []

        rules = {}

        def encode_non_terminal(current_type, context, depth):
            if len(context) == 0:
                return current_type, None, depth
            if n_gram == 2:
                return current_type, context[0], depth
            return current_type, context, depth

        list_to_be_treated = deque()
        list_to_be_treated.append((return_type, [], 0))

        counter = 0
        while len(list_to_be_treated) > 0:
            counter +=1 
            if counter % 5 == 0: print(f"iteration {counter}, deque len = {len(list_to_be_treated)}, rules written {len(rules)}")
            current_type, context, depth = list_to_be_treated.pop()
            non_terminal = encode_non_terminal(current_type, context, depth)

            # a non-terminal is a triple (type, context, depth)
            # if n_gram == 1 then context = None
            # otherwise context is a list of (primitive, number_argument)
            # print("\ncollecting from the non-terminal ", non_terminal)

            if non_terminal not in rules:
                rules[non_terminal] = {}

            # If program has the right deplth variables are possible expansions (no variables before min depth)
            if depth < max_program_depth and depth >= min_variable_depth:
                for i in range(len(args)):
                    if current_type == args[i]:
                        var = Variable(i, current_type, probability={})
                        rules[non_terminal][var] = []

            # At the end you all compatible constants
            if depth == max_program_depth - 1:
                for P in self.list_primitives:
                    type_P = P.type
                    return_P = type_P.returns()
                    if return_P == current_type and len(type_P.arguments()) == 0:
                        rules[non_terminal][P] = []

            elif depth < max_program_depth: # FC: looks like a slight error: < min_variable_depth? No, it is not error.
                for P in self.list_primitives:
                    if isinstance(P, BasicPrimitive) and P.primitive in self.no_repetitions and \
                            context and len(context) > 0 and context[0][0].primitive == P.primitive:
                        continue
                    type_P = P.type
                    # here is critical: ends_with must be aware of Union construct: if current_type = Union(A,B) and type_P in {A,B} then arguments_P != None
                    arguments_P = type_P.ends_with(current_type)
                    if arguments_P != None:
                        decorated_arguments_P = []
                        for i, arg in enumerate(arguments_P):
                            new_context = context.copy()
                            new_context = [(P, i)] + new_context
                            if len(new_context) > n_gram - 1:
                                new_context.pop()
                            decorated_arguments_P.append(
                                encode_non_terminal(arg, new_context, depth + 1)
                            )
                            if (arg, new_context, depth + 1) not in list_to_be_treated:
                                list_to_be_treated.appendleft(
                                    (arg, new_context, depth + 1)
                                )

                        rules[non_terminal][P] = decorated_arguments_P

        return ARC_CFG(
            start=(return_type, None, 0),
            rules=rules,
            max_program_depth=max_program_depth,
            clean=True
        )
    def ARC_DSL_to_ARC_CFG_fast(
        self,
        type_request,
        upper_bound_type_size=10,
        max_program_depth=4,
        min_variable_depth=1,
        n_gram=2,
    ):
        """
        Constructs a CFG from a DSL imposing bounds on size of the types
        and on the maximum program depth
        """
        self.instantiate_polymorphic_types()

        if isinstance(type_request, Arrow):
            return_type = type_request.returns()
            args = type_request.arguments()
        else:
            return_type = type_request
            args = []

        rules = {}

        def encode_non_terminal(current_type, context, depth):
            if len(context) == 0:
                return current_type, None, depth
            if n_gram == 2:
                return current_type, context[0], depth
            return current_type, context, depth
        def get_new_context(old_context, program, index):
            new_context = old_context.copy()
            new_context = [(program, index)] + new_context
            if len(new_context) > n_gram - 1:
                new_context.pop()
            return new_context

        list_to_be_treated = deque()
        list_to_be_treated.append((return_type, [], 0))

        counter = 0
        while len(list_to_be_treated) > 0:
            counter +=1 
            if counter % 10 == 0: print(f"iteration {counter}, deque len = {len(list_to_be_treated)}, rules written {len(rules)}, non terminals hashed {len(self.hashed_type_expansions)}")
            current_type, context, depth = list_to_be_treated.pop()
            non_terminal = encode_non_terminal(current_type, context, depth)

            # a non-terminal is a triple (type, context, depth)
            # if n_gram == 1 then context = None
            # otherwise context is a list of (primitive, number_argument)
            # print("\ncollecting from the non-terminal ", non_terminal)

            if non_terminal not in rules:
                rules[non_terminal] = {}

            if depth < max_program_depth and depth >= min_variable_depth:
                for i in range(len(args)):
                    if current_type == args[i]:
                        var = Variable(i, current_type, probability={})
                        rules[non_terminal][var] = []

            if depth == max_program_depth - 1:
                for P in self.list_primitives:
                    type_P = P.type
                    return_P = type_P.returns()
                    if return_P == current_type and len(type_P.arguments()) == 0:
                        rules[non_terminal][P] = []

            elif depth < max_program_depth:
                if current_type.hash in self.hashed_type_expansions:
                    P2arguments_dict=self.hashed_type_expansions[current_type.hash]
                    for P,arguments_P in P2arguments_dict.items():
                        decorated_arguments_P=[encode_non_terminal(arg, get_new_context(context,P,i), depth+1) for i,arg in enumerate(arguments_P)]
                        rules[non_terminal].update({P:decorated_arguments_P})
                        list_to_be_treated.extendleft((arg, get_new_context(context,P,i), depth + 1) for i,arg in enumerate(arguments_P) if (arg, get_new_context(context,P,i), depth + 1) not in list_to_be_treated)
                        # for i,arg in enumerate(arguments_P):
                        #     new_context = get_new_context(context,P,i)
                        #     if (arg, new_context, depth + 1) not in list_to_be_treated: 
                        #         list_to_be_treated.appendleft( (arg, new_context, depth + 1))
                else:
                    self.hashed_type_expansions[current_type.hash]={}
                    for P in self.list_primitives:
                        if isinstance(P, BasicPrimitive) and P.primitive in self.no_repetitions and \
                                context and len(context) > 0 and context[0][0].primitive == P.primitive:
                            continue
                        type_P = P.type
                        arguments_P = type_P.ends_with(current_type)
                        if arguments_P != None:
                            decorated_arguments_P = []
                            for i, arg in enumerate(arguments_P):
                                new_context = get_new_context(context,P,i)
                                decorated_arguments_P.append(
                                    encode_non_terminal(arg, new_context, depth + 1)
                                )
                                if (arg, new_context, depth + 1) not in list_to_be_treated:
                                    list_to_be_treated.appendleft(
                                        (arg, new_context, depth + 1)
                                    )
                            rules[non_terminal][P] = decorated_arguments_P
                            self.hashed_type_expansions[current_type.hash].update({P:arguments_P})

        return ARC_CFG(
            start=(return_type, None, 0),
            rules=rules,
            max_program_depth=max_program_depth,
            clean=True
        )

