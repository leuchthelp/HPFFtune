import inspect
import types
from typing import cast


def get_calling_func_name() -> str:
    this = cast(types.FrameType, inspect.currentframe())
    parent = cast(types.FrameType, this.f_back)

    return parent.f_code.co_name
