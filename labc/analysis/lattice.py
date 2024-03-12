
import copy

RBC = {
    'C1': {
        'a': 0.011, # fm
        'ainv': 1.79, # GeV
        'L': 24,
        'T': 64,
    }
}

E300 = {
    'L': 96,
    'T': 192,
    'a': None,
    'beta': 3.7,
}

# class Lattice:

#     def __init__(self, ensemble):
#         self.ensemble = ensemble
#         self._ensemble_dict = RBC[ensemble]
#         for key, value in self._ensemble_dict.items():
#             setattr(self, key, value)
    
#     def __str__(self):
#         out = 30*"#" + "\n"
#         out += "# Lattice\n"
#         out += 30*"#" + "\n"
#         out += f"# Ensemble '{self.ensemble}'\n"
#         for key, value in self._ensemble_dict.items():
#             out += f"#  {chr(183)}{key}: {value}\n"
#         out += "############"
#         return out
    

# class DatabaseEnsemble:
    
#     def __init__(self):
#         self._register_file = {}
#         self._register_func_raw = {}
#         self._register_func_stats = {}
#         self._register_func = {}


#     # def set_stats(self, statsType, tsrc_list):
#     #     self.statsType = statsType
#     #     self.tsrc_list = tsrc_list
#     #     self._apply_stats()
    
#     def _apply_stats(self, ensemble):
#         # for func in self._register_func_raw.values():
#         #     setattr(self, func.__name__, func(self.statsType, self.tsrc_list))
#         for func in self._register_func_stats.values():
#             setattr(self, func.__name__, func(ensemble))

#     # def __init__(self):
#     #     self._register = {}
    
#     # def add_file(self, file_func):
#     #     """Add function that returns the file name as an attribute to 
#     #     the data base."""

#     #     key = file_func.__name__
#     #     if not key in self._register.keys():
#     #         self._register_file[key] = file_func
#     #         setattr(self, file_func.__name__, file_func)
#     #     else:
#     #         raise ValueError(
#     #             message = f"Database file {file_func.__name__} already defined."
#     #         )
#     #     return file_func
            
#     # def add_func_raw(self, file_func):
#     #     if not file_func.__name__ in self._register.keys():
#     #         self._register_func_raw[file_func.__name__] = file_func
#     #     else:
#     #         raise ValueError(
#     #             message = f"Database function '{file_func}' already defined."
#     #         )
    
#     def add_func_stats(self, file_func):
#         if not file_func.__name__ in self._register_func_stats.keys():
#             self._register_func_stats[file_func.__name__] = file_func
#         else:
#             raise ValueError(
#                 message = f"Database function '{file_func}' already defined."
#             )
    
#     # def add_func(self, file_func):
#     #     if not file_func.__name__ in self._register.keys():
#     #         self._register_func[file_func.__name__] = file_func
#     #     else:
#     #         raise ValueError(
#     #             message = f"Database function '{file_func}' already defined."
#     #         )


class Ensemble:

    def __init__(self, ID: str, info: dict, statsType, tsrc_list):
        self.ID = ID
        self.info = info
        self.statsType = statsType
        self.tsrc_list = tsrc_list
        self.globals = {}

    def add_globals(self, key, value):
        self.globals[key] = value

    def add_database(self, databaseEnsemble):
        database = databaseEnsemble
        database._apply_stats(self)
        self.data = database
