# Main version
__version__ = "0.1.0"
# Main license
__license__ = "MIT"

def _import_version_metadata(mod):
    import importlib

    target_module = importlib.import_module(mod, package="aiwonderland")
    try:
        sub_module_version = target_module.__version__
    except AttributeError:
        raise SyntaxError("The submodule {mod} does not declare the __version__ version field") from None

    return sub_module_version

__functools_version__ = _import_version_metadata(".functools")
