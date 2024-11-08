import os
import sys
import re

from ARC_type_system import Type, PolymorphicType, PrimitiveType, Arrow, UnknownType
from cons_list import index

import ARC_constants
ARC_constants_names = [item for item in dir(ARC_constants) if not item.startswith("__")]


from time import perf_counter
from itertools import combinations_with_replacement

# dictionary { number of environment : value }
# environment: a cons list
# list = None | (value, list)
# probability: a dictionary {(G.__hash(), S) : probability}
# such that P.probability[S] is the probability that P is generated
# from the non-terminal S when the underlying PCFG is G.

# make sure hash is deterministic
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

class Program:
    """
    Object that represents a program: a lambda term with basic primitives
    """

    def __eq__(self, other):
        return (
            isinstance(self, Program)
            and isinstance(other, Program)
            and self.type.__eq__(other.type)
            and self.typeless_eq(other)
        )

    def typeless_eq(self, other):
        b = isinstance(self, Program) and isinstance(other, Program)
        b2 = (
            isinstance(self, Variable)
            and isinstance(other, Variable)
            and self.variable == other.variable
        )
        b2 = b2 or (
            isinstance(self, Function)
            and isinstance(other, Function)
            and self.function.typeless_eq(other.function)
            and len(self.arguments) == len(other.arguments)
            and all(
                [
                    x.typeless_eq(y)
                    for x, y in zip(self.arguments, other.arguments)
                ]
            )
        )
        b2 = b2 or (
            isinstance(self, Lambda)
            and isinstance(other, Lambda)
            and self.body.typeless_eq(other.body)
        )
        b2 = b2 or (
            isinstance(self, BasicPrimitive)
            and isinstance(other, BasicPrimitive)
            and self.primitive == other.primitive
        )
        b2 = b2 or (
            isinstance(self, New)
            and isinstance(other, New)
            and (self.body).typeless_eq(other.body)
        )
        return b and b2

    def __gt__(self, other):
        True

    def __lt__(self, other):
        False

    def __ge__(self, other):
        True

    def __le__(self, other):
        False

    def __hash__(self):
        return self.hash

    def is_constant(self):
        return True

    def derive_with_constants(self, constants):
        return self

    def make_all_constant_variations(self, constants_list):
        n_constants = self.count_constants()
        if n_constants == 0:
            return [self]
        all_possibilities = combinations_with_replacement(constants_list, n_constants)
        return [self.derive_with_constants(list(possibility)) for possibility in all_possibilities]
        

    def count_constants(self):
        return 0

class Variable(Program):
    def __init__(self, variable, type_=UnknownType(), probability={}):
        # assert isinstance(variable, int)
        self.variable = variable
        # assert isinstance(type_, Type)
        self.type = type_
        self.hash = variable

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        return "( var" + format(self.variable) + " )"

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            # logging.debug('Already evaluated')
            return self.evaluation[i]
        # logging.debug('Not yet evaluated')
        try:
            result = index(environment, self.variable)
            self.evaluation[i] = result
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = index(environment, self.variable)
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            return None

    def is_constant(self):
        return False

class Function(Program):
    def __init__(self, function, arguments, type_=UnknownType(), probability={}):
        # assert isinstance(function, Program)
        self.function = function
        # assert isinstance(arguments, list)
        self.arguments = arguments
        self.type = type_
        self.hash = hash(tuple([arg.hash for arg in self.arguments] + [self.function.hash]))
        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        if len(self.arguments) == 0:
            return "( " + format(self.function) + " )"
        else:
            s = "( " + format(self.function)
            for arg in self.arguments:
                s += " " + format(arg)
            return s + " )"

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            return self.evaluation[i]
        try:
            if len(self.arguments) == 0:
                return self.function.eval(dsl, environment, i)
            else:
                evaluated_arguments = []
                for j in range(len(self.arguments)):
                    e = self.arguments[j].eval(dsl, environment, i)
                    evaluated_arguments.append(e)
                result = self.function.eval(dsl, environment, i)
                for evaluated_arg in evaluated_arguments:
                    result = result(evaluated_arg)
                self.evaluation[i] = result
                return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError, StopIteration, RuntimeError, ZeroDivisionError):
            self.evaluation[i] = None
            return None
        except RecursionError: 
            print(self)
            return None

    def eval_naive(self, dsl, environment):
        st = perf_counter()
        try:
            if len(self.arguments) == 0:
                en = perf_counter()
                if en-st > 1: raise TimeoutError(f"timeout with {en-st} time for {self}, {environment}") 
                return self.function.eval_naive(dsl, environment)
            else:
                evaluated_arguments = []
                for j in range(len(self.arguments)):
                    e = self.arguments[j].eval_naive(dsl, environment)
                    evaluated_arguments.append(e)
                en = perf_counter()
                if en-st > 1: raise TimeoutError(f"timeout with {en-st} time for {self}, {environment}") 
                result = self.function.eval_naive(dsl, environment)
                for evaluated_arg in evaluated_arguments:
                    result = result(evaluated_arg)
                en = perf_counter()
                if en-st > 1: raise TimeoutError(f"timeout with {en-st} time for {self}, {environment}") 
                return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError, StopIteration, RuntimeError, ZeroDivisionError, TimeoutError):
            return None
        except RecursionError: 
            print(self)
            return None

    def is_constant(self):
        return all([self.function.is_constant()] + [arg.is_constant() for arg in self.arguments])

    def count_constants(self):
        return self.function.count_constants() + sum([arg.count_constants() for arg in self.arguments])

    def derive_with_constants(self, constants):
        return Function(self.function.derive_with_constants(constants), [argument.derive_with_constants(constants) for argument in self.arguments], self.type, self.probability)

class Lambda(Program):
    def __init__(self, body, type_=UnknownType(), probability={}):
        # assert isinstance(body, Program)
        self.body = body
        # assert isinstance(type_, Type)
        self.type = type_
        self.hash = hash(94135 + body.hash)

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        s = "( lambda " + format(self.body) + " )"
        return s

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            # logging.debug('Already evaluated')
            return self.evaluation[i]
        # logging.debug('Not yet evaluated')
        try:
            result = lambda x: self.body.eval(dsl, (x, environment), i)
            self.evaluation[i] = result
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = lambda x: self.body.eval_naive(dsl, (x, environment))
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            return None

class BasicPrimitive(Program):
    def __init__(self, primitive, type_=UnknownType(), probability={}, constant_evaluation=None):
        # assert isinstance(primitive, str)
        self.primitive = primitive
        # assert isinstance(type_, Type)
        self.type = type_
        self.is_a_constant = not isinstance(type_, Arrow)# and primitive.startswith("constant") # FC: no constants otherwise
        self.constant_evaluation = constant_evaluation
        self.hash = hash(primitive) + self.type.hash

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        """
        representation without type
        """
        if self.is_a_constant and self.constant_evaluation:
            return "( " + format(self.constant_evaluation) + " )"
        if self.is_a_constant:
            return "( " + format(self.primitive) + " )"
        if self.primitive in ARC_constants_names:
            return "( " + format(self.primitive) + " )"
        return format(self.primitive)

    def eval(self, dsl, environment, i):
        if self.is_a_constant and self.constant_evaluation:
            return self.constant_evaluation
        return dsl.semantics[self.primitive]

    def eval_naive(self, dsl, environment):
        if self.is_a_constant and self.constant_evaluation:
            return self.constant_evaluation
        return dsl.semantics[self.primitive]

    def count_constants(self):
        return 1 if self.is_a_constant else 0

    def derive_with_constants(self, constants):
        if self.is_a_constant:
            return BasicPrimitive(self.primitive, self.type, self.probability, constants.pop())
        else:
            return self


class New(Program):
    def __init__(self, body, type_=UnknownType(), probability={}):
        self.body = body
        self.type = type_
        self.hash = hash(783712 + body.hash) + type_.hash

        self.probability = probability
        self.evaluation = {}

    def __repr__(self):
        return format(self.body)

    def eval(self, dsl, environment, i):
        if i in self.evaluation:
            # logging.debug('Already evaluated')
            return self.evaluation[i]
        # logging.debug('Not yet evaluated')
        try:
            result = self.body.eval(dsl, environment, i)
            self.evaluation[i] = result
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            self.evaluation[i] = None
            return None

    def eval_naive(self, dsl, environment):
        try:
            result = self.body.eval_naive(dsl, environment)
            return result
        except (AttributeError, IndexError, ValueError, OverflowError, TypeError):
            return None

    def is_constant(self):
        return self.body.is_constant()

    def count_constants(self):
        return self.body.count_constants()

    def derive_with_constants(self, constants):
        return New(self.body.derive_with_constants(constants), self.type, self.probability)


def string2function(ast_str: str, types) -> Function:
    # Remove leading/trailing whitespace and verify proper parenthesis
    ast_str = ast_str.strip()
    if not ast_str.startswith("(") or not ast_str.endswith(")"):
        raise ValueError("Invalid AST format. Missing parentheses.")

    def parse_tokens(tokens: list[str]):
        # Parse tokens into an ASTNode recursively
        token = tokens.pop(0)
        if token == "(":
            # Start a new function node
            func_name = tokens.pop(0)  # The function name
            node = Function(function = BasicPrimitive(func_name, type_ = types[func_name]), arguments = [])
            while tokens[0] != ")":
                if tokens[0] == "(":
                    node.arguments.append(parse_tokens(tokens))
                else:
                    # Base case for literals or variable names
                    name = tokens.pop(0)
                    if name.startswith("var"):
                        node.arguments.append(Variable(variable = name[3:]))
                    else:
                        node.arguments.append(BasicPrimitive(name, type_ = types[name]))
            tokens.pop(0)  # Remove closing parenthesis
            return node
        elif token == ")":
            raise ValueError("Unexpected closing parenthesis")
        else:
            print("???")
        # else:
        #     print(1)
        #     name = tokens.pop(0)
        #     if name.startswith("var"):
        #         return Variable(variable = name[3:])
        #     else:
        #         return BasicPrimitive(name, type_ = types[name])

    # Tokenize the input string
    tokens = re.findall(r"\(|\)|\w+", ast_str)
    return parse_tokens(tokens)

# ast_str = "repeat"
# ast_str = "SEVEN"
# ast_str = "(repeat (leastcommon_ct var0) SEVEN)"
# ast_str = "(downscale (apply_ct (repeat FIVE) (mostcommon_ct var0)) (size_ct (bottomhalf var0)))"
# ast_str = "(switch (leastcommon_cf (initset var0)) SEVEN (valmax_ct (repeat THREE_BY_THREE SIX) (index var0)))"
# import sys
# sys.path.append(r"C:\Users\Francesco\Desktop\github_repos\ARC\code\auxillary_github_repos\DeepSynth\DSL")
# from DSL.ARC_formatted_dsl import primitive_types
# print(ast_str)
# program = string2function(ast_str, types = primitive_types)
# print(program)

def combine_programs_naive(*args, dsl, environment):
    if len(environment)>0:
        return TypeError("Multiple arguments not for this case")
    inp = environment[0] 
    programs: list[Function] = args
    for program in programs:
        out = program.eval_naive(dsl, [inp])
        inp = out
    return out
def combine_programs(*args, dsl, environment, i):
    if len(environment)>0:
        return TypeError("Multiple arguments not for this case")
    inp = environment[0] 
    programs: list[Function] = args
    for program in programs:
        out = program.eval(dsl, [inp], i)
        inp = out
    return out

def format_program_full(program, types, level = 0)->str:    
    if isinstance(program, BasicPrimitive):
        return format(program) + " type: " + str(types[format(program)]) + "\n"
    if isinstance(program, Variable):
        return format(program) + " type: GRID\n"
    if len(program.arguments) == 0:
        return format(program.function) + " type: " + str(types[format(program.function)]) + "\n"
    else:
        s = format(program.function) + " type: " + str(types[format(program.function)]) + "\n"
        for arg in program.arguments:
            s += "\t"*(level+1) + format_program_full(arg, types, level = level+1)
        return s