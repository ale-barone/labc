import pathlib
import inspect

def _get_inner_classes(cls):
    return [inner for inner in dir(cls) 
            if inspect.isclass(getattr(cls, inner))
            and not inner == '__class__']

def _get_extension(file):
    path = pathlib.Path(file)
    return path.suffix

class objectFactory:

    # do I actually need this? maybe also in class obj?
    def __init__(self):
        self._creators = {}
        self._extensions = set(self._creators.keys()) # maybe I should just use _creators.keys()
        self._statsID = {str(ext): _get_inner_classes(obj) for ext, obj in self._creators.items()}
    
    def add_obj(self, extension, obj):
        if extension in self._extensions:
            raise KeyError(f"Extension {extension} already defined.")
        else:
            self._creators[extension] = obj
            self._extensions.add(extension)
            self._statsID[extension] = _get_inner_classes(obj) # careful here, I may want to 

    def get_obj(self, file, statsID):
        extension = _get_extension(file)
        if not extension in self._extensions:
            raise ValueError(f"Unknown extension '{extension}'")

        def get_obj_inner(obj, statsID):
            list_id = _get_inner_classes(obj)
            if not statsID in list_id:
                raise ValueError(f"statsID '{statsID}' is not available for class '{obj.__name__}'.\n"
                                 f"Available options are {list_id}.") # TODO: implement nicer printing
            obj_inner = getattr(obj, statsID)
            return obj_inner

        obj = get_obj_inner(self._creators[extension], statsID)
        return obj