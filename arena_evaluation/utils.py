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