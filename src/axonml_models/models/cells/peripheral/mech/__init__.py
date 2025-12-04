import os
import importlib

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
        if class_name.lower() in name.lower():
            class_name = name
            globals()[name] = getattr(module, name)  # Add class to global namespace
            _registry[name] = getattr(module, name)  # Add to registry


def get_mechanism(name):
    return _registry.get(name, None)

get_mechanism.available = lambda: list(_registry.keys())
