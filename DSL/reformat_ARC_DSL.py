import ARC_dsl
import ARC_constants
from ARC_types import *
import types

#print(ARC_dsl.__dict__)
# for k,v in ARC_dsl.__dict__.items():
#     print(k,'----->',type(v), 'is function?', isinstance(v, types.FunctionType))

def curry123(f: Callable):
    n_args = f.__code__.co_argcount
    if n_args == 1:
        return f
    elif n_args == 2:
        return lambda x: lambda y: f(x, y)
    elif n_args == 3:
        return lambda x: lambda y: lambda z: f(x, y, z)
    elif n_args == 4:
        assert f.__name__ == "objects", f"Only quaternary function is objects! Found instead {f.__name__}"
        return lambda x: lambda y: lambda z: lambda w: f(x, y, z, w)
    else:
        string = f"Function with too many arguments! function name: {f.__name__}"
        print(">>>>>>>>>>>>>>>>>>>", string)
        raise TypeError("Look above!")


functions = {k:curry123(v) for k,v in ARC_dsl.__dict__.items() if isinstance(v, types.FunctionType)}
uncarried_functions = {k:v for k,v in ARC_dsl.__dict__.items() if isinstance(v, types.FunctionType)}


constants = {k:v for k,v in ARC_constants.__dict__.items() if type(v) in [bool, int, tuple]}

semantics = {}
semantics.update(functions)
semantics.update(constants)

from ARC_type_system import INT, INTEGER_TUPLE
constant2type = {
    "F": INT,
    "T": INT,
    "ZERO": INT,
    "ONE": INT,
    "TWO": INT,
    "THREE": INT,
    "FOUR": INT,
    "FIVE": INT,
    "SIX": INT,
    "SEVEN": INT,
    "EIGHT": INT,
    "NINE": INT,
    "TEN": INT,
    "NEG_ONE": INT,
    "NEG_TWO": INT,
    "DOWN": INTEGER_TUPLE,
    "RIGHT": INTEGER_TUPLE,
    "UP":  INTEGER_TUPLE,
    "LEFT":  INTEGER_TUPLE,
    "ORIGIN": INTEGER_TUPLE,
    "UNITY": INTEGER_TUPLE,
    "NEG_UNITY": INTEGER_TUPLE,
    "UP_RIGHT":  INTEGER_TUPLE,
    "DOWN_LEFT":  INTEGER_TUPLE,
    "ZERO_BY_TWO": INTEGER_TUPLE,
    "TWO_BY_ZERO": INTEGER_TUPLE,
    "TWO_BY_TWO": INTEGER_TUPLE,
    "THREE_BY_THREE": INTEGER_TUPLE,
}

from inspect import signature

def format_type(tt)->str:
    if tt == int:
        return "INT"
    if tt == bool:
        return "BOOL"
    return  str(tt).replace("typing.","").replace("[","(").replace("]",")").replace("int", "INT").replace("bool", "BOOL")

def write_recursively_types():
    fo=open("DSL\\ARC_formatted_dsl.py","w")
    fo.write("""from ARC_type_system import *

t0 = PolymorphicType('t0')
t1 = PolymorphicType('t1')
t2 = PolymorphicType('t2')
t3 = PolymorphicType('t3')
t4 = PolymorphicType('t4')
#a0 = ExceedinglyPolymorphicType('a0') # TODO: for now it is not used
z0 = PolymorphicTypeOrPrimitiveArrow('z0')
""")
    fo.write("\n\nprimitive_types = {\n")
    for k,f in uncarried_functions.items():
        arg_type_names = [format_type(v.annotation) for k,v in dict(signature(f).parameters).items()]
        ret_type_name = format_type(f.__annotations__["return"])
        to_write = f"\t\"{f.__name__}\": Arrow({arg_type_names[0]},YYY),\n"
        for arg in arg_type_names[1:]:
            to_write = to_write.replace("YYY",f"Arrow({arg},YYY)")
        to_write = to_write.replace("YYY",ret_type_name)
        fo.write(to_write)
    for c_name,c_type in constant2type.items():
        fo.write(f"\t\"{c_name}\": {c_type},\n")
    fo.write("}")
    fo.close()
def write_constructed_types():
    import os
    os.environ["PRINT_CONSTRUCTED_TYPES"] = "True"
    write_recursively_types()

write_constructed_types()

from ARC_formatted_dsl import primitive_types as primitive_types_with_constants

no_repetitions = {}

# import sys
# sys.path.append("C:\\Users\\Francesco\\Desktop\\github_repos\\ARC\\code\\auxillary_github_repos\\DeepSynth")
from ARC_dsl_cfg import ARC_DSL
# from ..dsl import dsl # does not work
inst_ARC_dsl = ARC_DSL(primitive_types=primitive_types_with_constants,
                       semantics=semantics,
                       no_repetitions=no_repetitions)

from ARC_type_system import Arrow, GRID
#inst_ARC_cfg = inst_ARC_dsl.ARC_DSL_to_ARC_CFG(type_request=Arrow[GRID,GRID])
# fo = open("compiled_ARC_dsl.txt", "w")
# fo.write(str(inst_ARC_dsl))
# fo.close()

print(inst_ARC_dsl.primitive_types())
# print(inst_ARC_dsl.return_types())
# print(inst_ARC_dsl.all_type_requests(0))
# print(inst_ARC_dsl.all_type_requests(1))
# print(inst_ARC_dsl.all_type_requests(2)) # ouch!
# print(len(inst_ARC_dsl.all_type_requests(0)))
# print(len(inst_ARC_dsl.all_type_requests(1)))
# print(len(inst_ARC_dsl.all_type_requests(2)))

fo = open("DSL\\compiled_ARC_dsl_pre_poly.txt", "w")
print(f"Attento, sto per scrivere una stringa lunga {len(str(inst_ARC_dsl))}")
fo.write(str(inst_ARC_dsl))
fo.close()

inst_ARC_dsl.instantiate_polymorphic_types()

fo = open("DSL\\compiled_ARC_dsl_post_poly.txt", "w")
print(f"Attento, sto per scrivere una stringa lunga {len(str(inst_ARC_dsl))}")
fo.write(str(inst_ARC_dsl))
fo.close()

print("Fiuuuuuuuu")


# print(semantics)

# print(constants)
# print(constants['T'])


# print(semantics["trim"])
# print(semantics["identity"])
# print((id:=semantics["identity"])([[1,2],[3,4]]))