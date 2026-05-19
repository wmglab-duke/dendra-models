import os
import importlib

from dendra.models.mechanisms import Mechanism

# Get the directory of the current module
module_dir = os.path.dirname(__file__)

# List all Python files in the directory (excluding __init__.py)
py_files = [
    f for f in os.listdir(module_dir) if f.endswith(".py") and f != "__init__.py"
]

_registry = {}

# Import each file and fetch its classes
for file in py_files:
    module_name = file[:-3]  # Remove '.py' extension
    module = importlib.import_module(
        f".{module_name}", package=__name__
    )  # Import as relative module
    class_name = (
        module_name  # Assume class name is the capitalized version of file name
    )
    all_names = dir(module)
    for name in all_names:
        if name.lower().startswith(module_name.lower()):
            c_ = getattr(module, name)
            if isinstance(c_, type) and issubclass(c_, Mechanism):
                globals()[name] = c_  # Add class to global namespace
                _registry[name] = c_  # Add to registry


def get_mechanism(name):
    return _registry.get(name, None)

get_mechanism.available = lambda: list(_registry.keys())
