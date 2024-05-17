from abc import ABC


class Node(ABC):
    pass


class CommonNode(Node):
    pass


class Identifier(CommonNode):
    def __init__(self, identifier: str):
        self.identifier = identifier


class Verbatim(CommonNode):
    def __init__(self, text: str):
        self.text = text


class String(CommonNode):
    def __init__(self, text: str):
        self.text = text


class Integer(CommonNode):
    def __init__(self, integer: int):
        self.integer = integer


class Float(CommonNode):
    def __init__(self, fl: float):
        self.fl = fl


class Bool(CommonNode):
    def __init__(self, b: bool):
        self.b = b


LiteralExpression = Verbatim | String | Integer | Float | Bool
