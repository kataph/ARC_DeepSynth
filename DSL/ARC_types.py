from typing import (
    List,
    Union,
    Tuple,
    Any,
    Container,
    Callable,
    FrozenSet,
    Iterable
)

Boolean = bool
Integer = int
IntegerTuple = Tuple[Integer, Integer]
Numerical = Union[Integer, IntegerTuple]
IntegerSet = FrozenSet[Integer]
Grid = Tuple[Tuple[Integer]]
Cell = Tuple[Integer, IntegerTuple]
Object = FrozenSet[Cell]
Objects = FrozenSet[Object]
Indices = FrozenSet[IntegerTuple]
IndicesSet = FrozenSet[Indices]
Patch = Union[Object, Indices]
Element = Union[Object, Grid]
Piece = Union[Grid, Patch]
TupleTuple = Tuple[Tuple]
ContainerContainer = Container[Container]


# Boolean.__str__ = lambda s: "BOOL"
# Integer.__str__ = lambda s: "INT"
IntegerTuple.__str__ = lambda s: "IntegerTuple"
Numerical.__str__ = lambda s: "Numerical"
IntegerSet.__str__ = lambda s: "IntegerSet"
Grid.__str__ = lambda s: "Grid"
Cell.__str__ = lambda s: "Cell"
Object.__str__ = lambda s: "Object"
Objects.__str__ = lambda s: "Objects"
Indices.__str__ = lambda s: "Indices"
IndicesSet.__str__ = lambda s: "IndicesSet"
Patch.__str__ = lambda s: "Patch"
Element.__str__ = lambda s: "Element"
Piece.__str__ = lambda s: "Piece"
TupleTuple.__str__ = lambda s: "TupleTuple"
ContainerContainer.__str__ = lambda s: "ContainerContainer"
