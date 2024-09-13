import itertools
import typing
import numpy as np
import ast

class Pedestrian(typing.NamedTuple):
    id: str
    type: str
    social_state: str
    position: typing.List[float]
    theta: float
    destination: typing.List[float]

class Utils:
    @staticmethod
    def string_to_float_list(d):
        if not d:
            return []

        return np.array(d.replace("[", "").replace("]", "").split(r", ")).astype(float)

    @classmethod
    def parse_pedsim(cls, entry: str) -> typing.List[Pedestrian]:
        return [Pedestrian(**ped) for ped in ast.literal_eval(entry)]
    

T = typing.TypeVar("T")
class MultiDict(typing.Generic[T]):

    Hash = int
    Dimension = str
    Index = str
    HardPoint = typing.Dict[Dimension, Index]
    EasyPoint = typing.Tuple[Index, ...]
    Point = typing.Union[HardPoint, EasyPoint]

    _dimensions: typing.Tuple[Dimension, ...]
    _dimmap: typing.Dict[str, typing.Dict[str, typing.Set[Hash]]]
    _hashgen: typing.Iterator[Hash]
    _entries: typing.Dict[Hash, T]

    @property
    def dimensions(self) -> typing.Tuple[Dimension, ...]:
        return self._dimensions
    
    def axis(self, dimension: Dimension) -> typing.Tuple[Index, ...]:
        return tuple(self._dimmap[dimension].keys())
    
    def _hash(self, point: HardPoint) -> Hash:
        return next(self._hashgen)

    def _point(self, point: Point) -> HardPoint:
        if isinstance(point, tuple):
            return dict(zip(self.dimensions, point))
        return point

    def __len__(self) -> int:
        return len(self._entries)

    def __init__(self, *dimensions: Dimension):
        assert len(set(dimensions)) == len(dimensions), "duplicate dimension detected"
        self._hashgen = itertools.count()
        self._dimensions = dimensions
        self._entries = dict()
        self._dimmap = {dimension:dict() for dimension in dimensions}

    def retrieve(self, point: Point) -> typing.Collection[T]:
        point = self._point(point)

        intersection = set()
        first = True
        for dim, index in point.items():
            target = self._dimmap[dim].get(index, set())
            if first:
                intersection = target.copy()
                first = False
            else:
                intersection.intersection_update(target)
                if not len(intersection): return []

        return [self._entries[index] for index in intersection]

    def slice(self, full=False, **kwargs) -> typing.List[typing.Tuple[HardPoint, typing.Collection[T]]]:
        
        dimensions = self.dimensions

        targets = (
            dict(zip(dimensions, target))
            for target
            in itertools.product(*[
                list(self._dimmap[dim].keys()) if dim not in kwargs else (kwargs[dim] if isinstance(kwargs[dim], list) else [kwargs[dim]])
                for dim
                in dimensions
            ])
        )
        
        return list(
            (k,v)
            for k,v
            in (
                (
                    {k:v for k,v in self._point(target).items() if full or k not in kwargs},
                    self.retrieve(target)
                )
                for target
                in targets
            )
            if full or len(v)
        )

    def insert(self, point: Point, value: T):
        point = self._point(point)

        hash = self._hash(point)

        for dim, index in point.items():
            self._dimmap[dim].setdefault(index, set())
            self._dimmap[dim][index].add(hash)

        self._entries[hash] = value
