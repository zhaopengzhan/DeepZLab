import importlib
import logging
import pkgutil
import warnings
from pathlib import Path
from typing import Optional

from mmengine.registry import Registry

logger = logging.getLogger(__name__)
DeepZData = Registry('DATALOADERS')

#
_pkg_dir = Path(__file__).resolve().parent
_prefix = __name__ + '.'
for m in pkgutil.walk_packages([str(_pkg_dir)], _prefix):
    importlib.import_module(m.name)


def list_dataloaders():
    print(DeepZData)
    return list(DeepZData.module_dict)


# deprecated
def build_dataloader1(name: str,
                      **kwargs):
    """
    name: 在 Registry 里注册的 key
    kwargs    : 直接透传给模型 __init__
    """
    warnings.simplefilter("default", DeprecationWarning)
    warnings.warn("function is deprecated", DeprecationWarning)

    cfg = dict(type=name, **kwargs)
    return DeepZData.build(cfg)


def build_dataloader(name: str, use_cfg: Optional[bool] = False, **kwargs):
    """
    Build a dataloader by name.
    If use_cfg is True and the registered class defines a build method, try to call it.
    kwargs are merged into cfg and passed to the build procedure.
    """
    cls = DeepZData.module_dict.get(name)

    if cls is None:
        raise KeyError(f"{name} not found in DeepZData registry")
    cfg = dict(type=name, **kwargs)
    if use_cfg == False and hasattr(cls, "build") and callable(getattr(cls, "build")):
        try:
            return cls.build()
        except TypeError:
            logger.exception("cls.build raised exception for %s, fallback to Registry.build", name)
            return DeepZData.build(cfg)
    else:
        return DeepZData.build(cfg)


