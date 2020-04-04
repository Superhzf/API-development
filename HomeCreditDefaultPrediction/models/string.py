# Standard build-in libraries
from enum import Enum
# Related third party libraries


class EstEnum(str, Enum):
    @classmethod
    def quoted_values_string(cls) -> str:
        return ",".join(["'" + item.value + "'" for item in cls])
