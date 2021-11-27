import h5py
import numpy as np
from abc import ABC, abstractmethod
import inspect
from .HDF5Utilities import *
from ..objectInterface import IReader


# I should probably put all these either in '_HDF5ReaderBase' or in some extra
# utility file, in case I need to share these functions with the writer/exporter class
# def _get_groups(h5file):
#     groups = []
#     with h5py.File(h5file, 'r') as hf:
#         def _append_groups(name, obj):
#             if isinstance(obj, h5py._hl.group.Group):
#                 groups.append(name) 
#         hf.visititems(_append_groups)
#     return groups

# def _get_datasets(h5file):
#     datasets = []
#     with h5py.File(h5file, 'r') as hf:
#         def _append_datasets(name, obj):
#             if isinstance(obj, h5py._hl.dataset.Dataset):
#                 datasets.append(name)
#         hf.visititems(_append_datasets)
#     return datasets

# def _get_paths(h5file):
#     groups = _get_groups(h5file)
#     datasets = _get_datasets(h5file)
#     out = np.append(groups, datasets)
#     return out

# def _get_extension(file):
#     path = pathlib.Path(file)
#     return path.suffix




# reader_class = formatReader
# statsID = {'generic', 'gauge', 'stats'}  


class _HDF5ReaderBase(IReader):
    def __init__(self, file):
        super().__init__(file)
        self.paths = get_paths(file)
        self.groups = get_groups(file)
        self.datasets = get_datasets(file)   
        self.statsID = get_statsID(file)
    
    def read(self, dset, re=None):
        assert(dset in self.datasets), f"dataset '{dset}' must be in datasets list (check 'datasets' attribute)"
        with h5py.File(self.file, 'r') as hf:
            out = np.asarray(hf[dset])[re]
        return out

    def dump(self, re=None):
        num_dsets = len(self.datasets)
        with h5py.File(self.file, 'r') as hf:
            dsets = np.array([hf[self.datasets[d]] for d in range(num_dsets)])
        out_dict = {self.datasets[d]: np.asarray(dsets[d])[re] for d in range(num_dsets)}
        return out_dict 

    def get_structure(self):
        # TODO: I should implement the tree representation here! It would be super nice
        return self.paths


class HDF5Reader: # just as a namespace, for organization
    class generic(_HDF5ReaderBase):
        """Generic reader for HDF5 files."""
        pass    

    class gauge(_HDF5ReaderBase):
        """
        Reader for HDF5 files containing raw gauge data structured as
        /corr/{t_0/{config_0, config_1, ...}, t_1/{config_0, config_1 ...}, ...}
        """

        # TODO: Not good, I need to keep also the generic reader!
        def read_config(self, tsrc, config, re=None):
            path = f'corr/t{tsrc}/config{config}'
            #assert(dset in self.datasets), "'dset' must be in datasets list (check 'datasets' attribute)"
            with h5py.File(self.file, 'r') as hf:
                out = np.asarray(hf[path])[re]
            return out

                   
    class stats(_HDF5ReaderBase):
        """
        Reader for HDF5 files containing resampled data structured as
        /corr/{mean{mean}, err{err}, bins{bin_0, bin_2, ...}}
        """
        
        AVAIL_STATS = {'mean', 'err', 'bins'}

        def read(self, stats):
            if stats in __class__.AVAIL_STATS:
                path = f'/{stats}/{stats}'
                with h5py.File(self.file, 'r') as hf:
                    out = np.asarray(hf[path])
            else:
                raise ValueError(f"stats value '{stats}' is not defined. Avaiblable stats are {__class__.AVAIL_STATS}")
            return out



            

# # # my factory: it takes care of deciding which implementation to use based on some specific parameter 
# class readerFactory:

#     # do I actually need this? maybe also in class reader?
#     def __init__(self):
#         self._readers = {'.h5': HDF5Reader}
#         self._ID = {str(ext): get_inner_classes(reader) for ext, reader in self._readers.items()}
    
#     def add_reader(self, extension, reader):
#         self._readers[extension] = reader
#         self._ID[extension] = get_inner_classes(reader) # careful here, I may want to 

#     def get_reader(self, file, statsID: str):
#         extension = get_extension(file)
#         if not extension in self._readers.keys():
#             raise ValueError(f"Unknown extension '{extension}'")

#         def get_reader_inner(reader, statsID):
#             list_id = get_inner_classes(reader)
#             if not statsID in list_id:
#                 raise ValueError(f"statsID '{statsID}' is not available for reader class '{reader.__name__}'.\n"
#                                  f"Available options are {list_id}.") # TODO: implement nicer printing
#             reader_id = getattr(reader, statsID)
#             return reader_id

#         reader = get_reader_inner(self._readers[extension], statsID)
#         return reader


# # final object to be called: it deals with files, whereas readerFactory deals directly only with readers (doesn't really know anything about files)
# class reader:
#     """
#     Class which take as input the file and its 'ID' (typically 'generic', 'gauge' or 'stats') and calls the appropriate
#     reader with the help of a reader factory.
#     """

#     # maybe I can put this outside the class...we will see
#     factory = readerFactory()

#     def __new__(cls,  file, statsID): # TODO: in principle I should get the file ID from the attribute! (if there is no attribute then I go with 'generic')
#         reader = cls.factory.get_reader(file, statsID)
#         cls.reader = reader
#         return reader(file)

#     # this need some extra thinkg
#     # @classmethod
#     # def add_reader(cls, ext, reader):
#     #     cls.factory.add_reader(ext, reader)
#     #     #return __class__.factory
 

        


        