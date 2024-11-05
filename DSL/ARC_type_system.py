import os
import sys
#from functools import lru_cache #TODO: check that it works out
# from collections import defaultdict

'''
Objective: define a type system.
A type can be either PolymorphicType, PrimitiveType, Arrow, Tuple, or FrozenSet, Union
'''

#TODO: FC: any addition to the type system needs to ensure working of comparison and the other functions.

try:
    PRINT_CONSTRUCTED_TYPES = bool(os.environ["PRINT_CONSTRUCTED_TYPES"])
except: 
    PRINT_CONSTRUCTED_TYPES = True
# make sure hash is deterministic
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    #TODO: FC: added a warning...
    print("ATTENTION: I AM GONNA CLOSE THIS PROCESS AND RESTART!!!")
    print("TEMPORARILY REPLACED WITH NOT RESTARTING")
    os.environ['PYTHONHASHSEED'] = '0'
    #os.execv(sys.executable, [sys.executable] + sys.argv)


class Type:
    '''
    Object that represents a type
    '''
    hashed_eq: dict[tuple[str,str]] = dict()
    hashed_le: dict[tuple[str,str]] = dict()
    hashed_endswith: dict[tuple[str,str]] = dict()
    def __eq__(self, other):
        '''
        Type equality
        '''
        if (self.hash, other.hash) in self.hashed_eq: return self.hashed_eq[(self.hash, other.hash)]
        assert not (isinstance(self,UnknownType) or isinstance(other,UnknownType)), "No unkows"
        if not (isinstance(self, Type) and isinstance(other, Type)): self.hashed_eq[(self.hash, other.hash)] = False; return False
        if isinstance(self,PolymorphicType) and isinstance(other,PolymorphicType) and self.name == other.name: self.hashed_eq[(self.hash, other.hash)] = True; return True
        if (isinstance(self,PrimitiveType) and isinstance(other,PrimitiveType) and self.type == other.type): self.hashed_eq[(self.hash, other.hash)] = True; return True
        if (isinstance(self,Arrow) and isinstance(other,Arrow) and self.type_in.__eq__(other.type_in) and self.type_out.__eq__(other.type_out)): self.hashed_eq[(self.hash, other.hash)] = True; return True
        if (isinstance(self,Tuple) and isinstance(other,Tuple) and self.type_elt.__eq__(other.type_elt)): self.hashed_eq[(self.hash, other.hash)] = True; return True
        if (isinstance(self,FrozenSet) and isinstance(other,FrozenSet) and self.type_elt.__eq__(other.type_elt)): self.hashed_eq[(self.hash, other.hash)] = True; return True
        if (isinstance(self,Union) and isinstance(other,Union) and self.type_left.__eq__(other.type_left) and self.type_right.__eq__(other.type_right)): self.hashed_eq[(self.hash, other.hash)] = True; return True
        if (isinstance(self,Couple) and isinstance(other,Couple) and self.type_left.__eq__(other.type_left) and self.type_right.__eq__(other.type_right)): self.hashed_eq[(self.hash, other.hash)] = True; return True

        self.hashed_eq[(self.hash, other.hash)] = False; return False

    def __gt__(self, other): True
    def __lt__(self, other): False
    def __ge__(self, other): True
    def __le__(self, other): 
        if (self.hash, other.hash) in self.hashed_le: return self.hashed_le[(self.hash, other.hash)]
        assert not (isinstance(self,UnknownType) or isinstance(other,UnknownType)), "No unkows"
        if not (isinstance(self, Type) and isinstance(other, Type)):
            self.hashed_le[(self.hash, other.hash)] = False
            return False
        if isinstance(self,PolymorphicType) and isinstance(other,PolymorphicType) and self.name == other.name:
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if (isinstance(self,PrimitiveType) and isinstance(other,PrimitiveType) and self.type == other.type):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if (isinstance(self,Arrow) and isinstance(other,Arrow) and self.type_in.__le__(other.type_in) and self.type_out.__le__(other.type_out)):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if (isinstance(self,Tuple) and isinstance(other,Tuple) and self.type_elt.__le__(other.type_elt)):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if (isinstance(self,FrozenSet) and isinstance(other,FrozenSet) and self.type_elt.__le__(other.type_elt)):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if (isinstance(self,Couple) and isinstance(other,Couple) and self.type_left.__le__(other.type_left) and self.type_right.__le__(other.type_right)):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if (isinstance(self,Union) and (not isinstance(other,Union)) and self.type_left.__le__(other) and self.type_right.__le__(other)):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if ((not isinstance(self, Union)) and isinstance(other,Union) and (self <= other.type_left or self <= other.type_right)):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        if (isinstance(self, Union) and isinstance(other,Union) and (self.type_left <= other.type_left or self.type_left <= other.type_right) and (self.type_right <= other.type_left or self.type_right <= other.type_right)):
            self.hashed_le[(self.hash, other.hash)] = True
            return True
        self.hashed_le[(self.hash, other.hash)] = False
        return False
        #raise TypeError(f"I don't know how to compare {self} and {other}")
    
    
    def __hash__(self):
        return self.hash

    def returns(self):
        if isinstance(self,Arrow):
            return self.type_out.returns()
        else:
            return self

    def arguments(self):
        if isinstance(self,Arrow):
            return [self.type_in] + self.type_out.arguments()
        else:
            return []

    def list_ground_types(self, polymorphic=False):
        if isinstance(self, Arrow):
            return self.type_in.list_ground_types(polymorphic) + self.type_out.list_ground_types(polymorphic)
        elif isinstance(self, (Tuple, FrozenSet)):
            base = self.type_elt.list_ground_types(polymorphic)
            if base:
                base.append(self)
            return base
        elif isinstance(self, (Union,Couple)):
            base = self.type_left.list_ground_types(polymorphic) + self.type_right.list_ground_types(polymorphic)
            if base:
                base.append(self)
            return base
        else:
            if not polymorphic and isinstance(self, PolymorphicType):
                return []
            return [self]

    def ends_with(self, other):
        '''
        Checks whether other is a suffix of self and returns the list of arguments

        Example: 
        self = Arrow(INT, Arrow(INT, INT))
        other = Arrow(INT, INT)
        ends_with(self, other) = [INT]

        self = Arrow(Arrow(INT, INT), Arrow(INT, INT))
        other = INT
        ends_with(self, other) = [Arrow(INT, INT), INT]

        and for unions?
        self = Arrow(INT, Arrow(INT, INDICES))   (PATCH = Union(OBJECTS, INDICES)); PIECE = Union(GRID, PATCH))
        self2 = Arrow(INT, Arrow(INT, PATCH)) 
        other = Arrow(INT, PIECE) --> PIECE accepts GRID, OBJECTS, INDICES
        other2 = Arrow(INT, PATCH) --> PATCH accepts OBJECTS, INDICES
        other3 = Arrow(INT, PIECE) --> OBJECTS accepts only OBJECTS
        ends_with(self, other) = [INT]
        ends_with(self, other2) = [INT]
        ends_with(self, other3) = []
        ends_with(self2, other) = [INT]
        ends_with(self2, other2) = [INT]
        ends_with(self2, other3) = []
        '''
        if (self.hash, other.hash) in self.hashed_endswith: 
            return self.hashed_endswith[(self.hash, other.hash)]
        else:
            output = self.ends_with_rec(other, [])
            self.hashed_endswith[(self.hash, other.hash)] = output
            return output

    def ends_with_rec(self, other, arguments_list):
        if self <= other:
            return arguments_list
        if isinstance(self, Arrow):
            arguments_list.append(self.type_in)
            return self.type_out.ends_with_rec(other, arguments_list)
        return None

    def size(self):
        if isinstance(self,(PrimitiveType,PolymorphicType)):
            return 1
        if isinstance(self,Arrow):
            return self.type_in.size() + self.type_out.size()
        if isinstance(self,(FrozenSet, Tuple)):
            return self.type_elt.size() + 1
        # if isinstance(self,(FrozenSet, Tuple)) and isinstance(self.type_elt,(PrimitiveType,PolymorphicType)):
        #     return 2
        # if isinstance(self,(FrozenSet, Tuple)) and isinstance(self.type_elt,(FrozenSet, Tuple)) \
        # and isinstance(self.type_elt.type_elt,(PrimitiveType,PolymorphicType)):
        #     return 3
        if isinstance(self,(Union,)):
            return max(self.type_left.size(), self.type_right.size())
        if isinstance(self,(Couple,)):
            return max(self.type_left.size(), self.type_right.size()) + 1
        # We do not want List(List(List(...)))
        return 100_000

    def find_polymorphic_types(self):
        set_types = set()
        return self.find_polymorphic_types_rec(set_types)

    def find_polymorphic_types_rec(self, set_types: set):
        if isinstance(self,PolymorphicType):
            if not self.name in set_types:
                set_types.add(self.name)
        if isinstance(self,Arrow):
            set_types = self.type_in.find_polymorphic_types_rec(set_types)
            set_types = self.type_out.find_polymorphic_types_rec(set_types)
        if isinstance(self,(FrozenSet, Tuple)):
            set_types = self.type_elt.find_polymorphic_types_rec(set_types)
        if isinstance(self,(Union,Couple)):
            set_types = self.type_left.find_polymorphic_types_rec(set_types) | self.type_right.find_polymorphic_types_rec(set_types)
            assert len(set_types)==0, "No polymorphic types in unions!"
            return set_types
        return set_types
    

    def decompose_type(self) -> tuple[set,set]:
        '''
        Finds the set of basic types and polymorphic types 
        '''
        set_basic_types = set()
        set_polymorphic_types = set()
        return self.decompose_type_rec(set_basic_types,set_polymorphic_types)

    def decompose_type_rec(self,set_basic_types,set_polymorphic_types) -> tuple[set,set]:
        if isinstance(self,PrimitiveType):
            set_basic_types.add(self)
        if isinstance(self,PolymorphicType):
            set_polymorphic_types.add(self)
        if isinstance(self,Arrow):
            self.type_in.decompose_type_rec(set_basic_types,set_polymorphic_types)
            self.type_out.decompose_type_rec(set_basic_types,set_polymorphic_types)
        if isinstance(self,(Tuple,FrozenSet)):
            self.type_elt.decompose_type_rec(set_basic_types,set_polymorphic_types)
        if isinstance(self,(Union,Couple)):
            self.type_left.decompose_type_rec(set_basic_types,set_polymorphic_types)
            self.type_right.decompose_type_rec(set_basic_types,set_polymorphic_types)
        return set_basic_types,set_polymorphic_types

    def unify(self, other):
        '''
        Checks whether self can be instantiated into other
        # and returns the least unifier as a dictionary {t : type}
        # mapping polymorphic types to types.

        IMPORTANT: We assume that other does not contain polymorphic types.

        Example: 
        * list(t0) can be instantiated into list(int) and the unifier is {t0 : int}
        * list(t0) -> list(t1) can be instantiated into list(int) -> list(bool) 
        and the unifier is {t0 : int, t1 : bool}
        * list(t0) -> list(t0) cannot be instantiated into list(int) -> list(bool) 
        '''
        dic = {}
        if self.unify_rec(other, dic):
            return True
        else:
            return False

    def unify_rec(self, other, dic):
        if isinstance(self,PolymorphicType):
            if self.name in dic:
                return dic[self.name] == other
            else:
                dic[self.name] = other
                return True
        if isinstance(self,PrimitiveType):
            return isinstance(other,PrimitiveType) and self.type == other.type
        if isinstance(self,Arrow):
            return isinstance(other,Arrow) and self.type_in.unify_rec(other.type_in, dic) and self.type_out.unify_rec(other.type_out, dic)
        if isinstance(self,Tuple):
            return isinstance(other,Tuple) and self.type_elt.unify_rec(other.type_elt, dic)
        if isinstance(self,FrozenSet):
            return isinstance(other,FrozenSet) and self.type_elt.unify_rec(other.type_elt, dic)
        if isinstance(self,Union):
            return isinstance(other,Union) and self.type_left.unify_rec(other.type_left, dic) and self.type_right.unify_rec(other.type_right, dic)
        if isinstance(self,Couple):
            return isinstance(other,Couple) and self.type_left.unify_rec(other.type_left, dic) and self.type_right.unify_rec(other.type_right, dic)

    def apply_unifier(self, dic):
        # # for debug
        # if "Arrow(Arrow(t0,t1),Arrow(Arrow(t1,t2),Arrow(t0,t2)))" in str(self):
        #     print("Flag!")
        if isinstance(self,PolymorphicType):
            if self.name in dic:
                return dic[self.name]
            else:
                return self
        if isinstance(self,PrimitiveType):
            return self
        if isinstance(self,Arrow):
            new_type_in = self.type_in.apply_unifier(dic)
            new_type_out = self.type_out.apply_unifier(dic)
            return Arrow(new_type_in, new_type_out)
        if isinstance(self,Tuple):
            new_type_elt = self.type_elt.apply_unifier(dic)
            return Tuple(new_type_elt)
        if isinstance(self,FrozenSet):
            new_type_elt = self.type_elt.apply_unifier(dic)
            return FrozenSet(new_type_elt)
        if isinstance(self,Union):
            new_type_left = self.type_left.apply_unifier(dic)
            new_type_right = self.type_right.apply_unifier(dic)
            assert new_type_right == self.type_right and new_type_left == self.type_left, "No polymorphic types in a Union!"
            return Union(new_type_left,new_type_right)
        if isinstance(self,Couple):
            new_type_left = self.type_left.apply_unifier(dic)
            new_type_right = self.type_right.apply_unifier(dic)
            # assert new_type_right == self.type_right and new_type_left == self.type_left, "No polymorphic types in a Couple!"
            # Found exactly one function with polymorphic couple. 
            if not (new_type_right == self.type_right and new_type_left == self.type_left):
                print(f"Polymorphic touple (e.g. for product_cf) found in {self}! Take care.")
            return Couple(new_type_left,new_type_right)
    
    def __class_getitem__(cls, item): #for silly hack to get good print
        return cls._get_child_dict(item)#[item]
    @classmethod
    def _get_child_dict(cls, item):
        # return cls(item) # {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}
        if not hasattr(item, '__iter__'):
            return cls(item) # {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}
        if hasattr(item, '__iter__'):
            return cls(*item) # {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}

class PolymorphicType(Type):
    def __init__(self, name):
        assert(isinstance(name,str))
        self.name = name
        self.hash = hash(name)
    def __repr__(self):
        return format(self.name)
    
class ExceedinglyPolymorphicType(PolymorphicType):
    def __init__(self, name):
        assert(isinstance(name,str))
        self.name = name
        self.hash = hash(name)
    def __repr__(self):
        return format(self.name)
    
class PolymorphicTypeOrPrimitiveArrow(PolymorphicType):
    def __init__(self, name):
        assert(isinstance(name,str))
        self.name = name
        self.hash = hash(name)
    def __repr__(self):
        return format(self.name)
    
class PolymorphicTypeNoArrow(PolymorphicType):
    def __init__(self, name):
        assert(isinstance(name,str))
        self.name = name
        self.hash = hash(name)
    def __repr__(self):
        return format(self.name)
    
    

class PrimitiveType(Type):
    def __init__(self, type_):
        assert(isinstance(type_,str))
        self.type = type_
        self.hash = hash(type_)
    def __repr__(self):
        return format(self.type)

class Arrow(Type):
    def __init__(self, type_in, type_out):
        assert(isinstance(type_in,Type))
        assert(isinstance(type_out,Type)), f"type out was {type_out}, which is not a Type"
        self.type_in = type_in
        self.type_out = type_out
        self.hash = hash((type_in.hash,type_out.hash))

    def __repr__(self):
        rep_in = format(self.type_in)
        rep_out = format(self.type_out)
        #return "({} -> {})".format(rep_in, rep_out) #need another representation for printing code
        return "Arrow({},{})".format(rep_in, rep_out)

class Tuple(Type):
    def __init__(self, _type):
        assert(isinstance(_type,Type))
        self.type_elt = _type
        self.hash = hash(55555 + _type.hash)
    def __repr__(self):
        if isinstance(self.type_elt,Arrow):
            # return "Tuple{}".format(self.type_elt)
            # TODO: needed for proper printing of rapply
            return "Tuple({})".format(self.type_elt)
        elif not PRINT_CONSTRUCTED_TYPES:
            return "Tuple({})".format(self.type_elt)
        elif PRINT_CONSTRUCTED_TYPES:
            if self == GRID: return "GRID"
            if self != GRID: return "Tuple({})".format(self.type_elt) 

class FrozenSet(Type):
    def __init__(self, _type):
        assert(isinstance(_type,Type))
        self.type_elt = _type
        self.hash = hash(44444 + _type.hash)
    def __repr__(self):
        if isinstance(self.type_elt,Arrow):
            # This was for FrozenSet(in -> out)
            # return "FrozenSet{}".format(self.type_elt)
            return "FrozenSet({})".format(self.type_elt)
        elif not PRINT_CONSTRUCTED_TYPES:
            return "FrozenSet({})".format(self.type_elt)
        elif PRINT_CONSTRUCTED_TYPES:
            if self == OBJECT: return "OBJECT"
            if self == OBJECTS: return "OBJECTS"
            if self == INDICES: return "INDICES"
            if self == INTEGER_SET: return "INTEGER_SET"
            return "FrozenSet({})".format(self.type_elt)
        
# t0 = PolymorphicType('t0')
# x = FrozenSet[t0]
# y = FrozenSet[FrozenSet[t0]]
# print(str(x), str(y))


class Union(Type):
    def __init__(self, _type_left, _type_right):
        assert(isinstance(_type_left,Type))
        assert(isinstance(_type_right,Type))
        # for ground_type in _type_left.list_ground_types():
        #     assert not ground_type in (Arrow,),"!" 
        self.type_left = _type_left
        self.type_right = _type_right
        self.hash = hash(77777 + _type_left.hash + _type_right.hash)
    def __repr__(self):
        if isinstance(self.type_left,Arrow) or isinstance(self.type_right,Arrow):
            raise TypeError("No Arrows in union types")
            return "Union{}".format(self.type_elt)
        elif not PRINT_CONSTRUCTED_TYPES:
            return "Union({},{})".format(self.type_left,self.type_right)
        elif PRINT_CONSTRUCTED_TYPES:
            #if self == NUMERICAL: return "NUMERICAL" # suppressed
            if self == PATCH: return "PATCH"
            if self == ELEMENT: return "ELEMENT"
            if self == PIECE: return "PIECE"
            return "Union({},{})".format(self.type_left,self.type_right)

class Couple(Type):
    def __init__(self, _type_left, _type_right):
        assert(isinstance(_type_left,Type))
        assert(isinstance(_type_right,Type))
        # for ground_type in _type_left.list_ground_types()+_type_right.list_ground_types():
        #     assert not ground_type in (Arrow,),"!" 
        self.type_left = _type_left
        self.type_right = _type_right
        self.hash = hash(777112 + _type_left.hash + _type_right.hash)
    def __repr__(self):
        if isinstance(self.type_left,Arrow) or isinstance(self.type_right,Arrow):
            # raise TypeError("No Arrows in couple types")
            # TODO: there is indeed a case that it is needed
            print(f"An arrow in a couple, take care: type_left, _right is {self.type_left, self.type_right}")
            # old formatting, only for (in -> out) style arrow formatting
            # if (...) return "Couple{}".format(self.type_left,self.type_right) else ...
            return "Couple({},{})".format(self.type_left,self.type_right)
        elif not PRINT_CONSTRUCTED_TYPES:
            return "Couple({},{})".format(self.type_left,self.type_right)
        elif PRINT_CONSTRUCTED_TYPES:
            if self == INTEGER_TUPLE: return "INTEGER_TUPLE"
            if self == CELL: return "CELL"
            return "Couple({},{})".format(self.type_left,self.type_right)


class UnknownType(Type):
    '''
    In case we need to define an unknown type
    '''
    def __init__(self):
        self.type = ""
        self.hash = 1984

    def __repr__(self):
        return "UnknownType"

# primitive types
INT = PrimitiveType('INT')
BOOL = PrimitiveType('BOOL')

# constructed types
# INTEGER_TUPLE = Tuple(INT, INT) #fixed len tuple!
INTEGER_TUPLE = Couple(INT, INT) #fixed len tuple!
#NUMERICAL = Union(INT, INTEGER_TUPLE) # suppressed
INTEGER_SET = FrozenSet(INT)
GRID = Tuple(Tuple(INT)) # variadic tuple!
# CELL = Tuple(INT, INTEGER_TUPLE) #dishomog. tuple! #fixed len tuple!
CELL = Couple(INT, INTEGER_TUPLE) #dishomog. tuple! #fixed len tuple!
OBJECT = FrozenSet(CELL)
OBJECTS = FrozenSet(OBJECT)
INDICES = FrozenSet(INTEGER_TUPLE)
INDICES_SET = FrozenSet(INDICES)
PATCH = Union(OBJECT, INDICES)
ELEMENT = Union(OBJECT, GRID)
PIECE = Union(GRID, PATCH)

assert Couple(INT,INT).__eq__(INTEGER_TUPLE)
assert FrozenSet(FrozenSet(Couple(INT, INT))).__eq__(INDICES_SET)
assert Union(FrozenSet(Couple(INT, Couple(INT,INT))), FrozenSet(Couple(INT, INT))) == PATCH

assert (x:=INTEGER_TUPLE.size()) == 2, f"Actual size is {x}"
# assert (x:=NUMERICAL.size()) == 2, f"Actual size is {x}"
assert (x:=INTEGER_SET.size()) == 2, f"Actual size is {x}"
assert (x:=GRID.size()) == 3, f"Actual size is {x}"
assert (x:=CELL.size()) == 3, f"Actual size is {x}"
assert (x:=OBJECT.size()) == 4, f"Actual size is {x}"
assert (x:=OBJECTS.size()) == 5, f"Actual size is {x}"
assert (x:=INDICES.size()) == 3, f"Actual size is {x}"
assert (x:=INDICES_SET.size()) == 4, f"Actual size is {x}"
assert (x:=PATCH.size()) == 4, f"Actual size is {x}"
assert (x:=ELEMENT.size()) == 4, f"Actual size is {x}"
assert (x:=PIECE.size()) == 4, f"Actual size is {x}"

Integer = INT
Boolean = BOOL
IntegerTuple = INTEGER_TUPLE
# Numerical = NUMERICAL
IntegerSet = INTEGER_SET
Grid = GRID
Cell = CELL
Object = OBJECT
Objects = OBJECTS
Indices = INDICES
IndicesSet = INDICES_SET
Patch = PATCH
Element = ELEMENT
Piece = PIECE

# x = Couple(INT,INT)
# print(x) # prints Couple(INT,INT)
# PRINT_CONSTRUCTED_TYPES = True
# print(x) # prints INTEGER_TUPLE



self = Arrow(INT, Arrow(INT, INDICES))  # (PATCH = Union(OBJECT, INDICES)); PIECE = Union(GRID, PATCH))
self2 = Arrow(INT, Arrow(INT, PATCH)) 
assert (INT <= INT)
assert (OBJECT <= PATCH)
assert not (PATCH <= OBJECT)
assert (PATCH <= PIECE)
assert (INDICES <= PIECE)
assert (Union(INT, BOOL) <= Union(INT, BOOL))
assert not (Union(INT, PATCH) <= Union(INT, BOOL))
assert not (Union(OBJECTS, INDICES) <= PATCH)
assert (Union(OBJECT, INDICES) <= PATCH)
assert (Union(INT, BOOL) <= Union(INT, Union(BOOL, PATCH)))
assert (INDICES <= PATCH)
assert (OBJECT <= PIECE)
assert not (PIECE <= OBJECT)
assert Arrow(INT, Arrow(INT, INDICES)) <= Arrow(INT, Arrow(INT, PATCH))
other = Arrow(INT, PIECE) #--> PIECE accepts GRID, OBJECTS, INDICES
other2 = Arrow(INT, PATCH) #--> PATCH accepts OBJECTS, INDICES
other3 = Arrow(INT, OBJECTS) #--> OBJECTS accepts only OBJECTS
assert (x:=self.ends_with(other)) == [INT], f"Found {x} insted"
assert (x:=self.ends_with(other2)) == [INT], f"Found {x} insted"
assert (x:=self.ends_with(other3)) == None, f"Found {x} insted"
assert (x:=self2.ends_with(other)) == [INT], f"Found {x} insted"
assert (x:=self2.ends_with(other2)) == [INT], f"Found {x} insted"
assert (x:=self2.ends_with(other3)) == None, f"Found {x} insted"