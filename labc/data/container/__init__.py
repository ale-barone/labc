from ._objectFactory import ObjectFactory as _ObjectFactory
from .HDF5.reader import Reader as _HDF5Reader
from .HDF5.writer import Writer as _HDF5Writer
from .HDF5.formatter import Formatter as _HDF5Formatter
from .HDF5.utilities import check_fileID, get_fileID


# I THINK IT WOULD MAKE MORE SENSE TO HAVE THIS ALL FILE INTO THE __init__.py of 'dataManager'

# I may have to build a fileStatsID method which takes its own reader/writer

# class fileStatsID:
    
#     def __init__(self, file, fileID):
#         self.file = file
#         self.fileID = fileID
#         self.reader = reader(file)
#         self.writer = writer(file, fileID)

################################################################################
# READER
################################################################################

_reader_factory = _ObjectFactory()
_reader_factory.add_obj('.h5', _HDF5Reader)

# final object to be called: it deals with files, whereas readerFactory deals 
# directly only with readers (doesn't really know anything about files)
class Reader:
    """
    Class which take as input the file, read its 'fileID' ('gauge' or 'stats', 
    'generic' if None) and calls the appropriate reader with the help of a 
    reader factory.
    """

    def __new__(cls,  file): # TODO: in principle I should get the file ID from the attribute! (if there is no attribute then I go with 'generic')
        fileID = get_fileID(file)
        reader = _reader_factory.get_obj(file, fileID)
        cls.reader = reader
        # build reader
        out_reader = reader(file)
        out_reader.fileID = fileID
        return out_reader


################################################################################
# WRITER
################################################################################

_writer_factory = _ObjectFactory()
_writer_factory.add_obj('.h5', _HDF5Writer)

class Writer:
    """
    Class which take as input the file and its 'ID' (typically 'generic',
    'gauge' or 'stats') and calls the appropriate writer with the help of a
    writer factory.
    """
    
    def __new__(cls, file, fileID):
        writer = _writer_factory.get_obj(file, fileID)
        cls.writer = writer
        return writer(file, fileID)


################################################################################
# FORMATTER
################################################################################

_formatter_factory = _ObjectFactory()
_formatter_factory.add_obj('.h5', _HDF5Formatter)

class Formatter:
    """
    Class which take an input file with 'fileID'={'gauge', 'stats'}
    and format the data in a numpy array 'data' such that
    data[0]: mean
    data[1]: err
    data[2:]: bins
    """
    def __new__(cls, file, statsType):
        check_fileID(file)
        fileID = get_fileID(file)
        formatter = _formatter_factory.get_obj(file, fileID)
        cls.reader = formatter
        return formatter(file, statsType)



# # TODO: write this thing more in general! This is NOT acceptable!
# class extractor:
    
#     def __init__(self, file_fun, *args):
#         def _file_fun(tsrc, config):
#             return file_fun(tsrc, config, *args)
#         self.file_fun = _file_fun

#     # not efficient, I call the reader every time in extract_all!
#     def extract(self, tsrc, config, path, re):
#         ext_reader = reader(self.file_fun(tsrc, config))
#         data = np.roll(np.array(ext_reader.read(path)[re]), -tsrc)  
#         return data          

    
#     def extract_all(self, tsrc_list, config_list, path, re):
#         data = np.array([])
#         for tsrc in tsrc_list:
#             for config in config_list:
#                 data_single = self.extract(tsrc, config, path, re)
#                 data = np.append(data, data_single)
                
#         num_sources = len(tsrc_list)
#         num_config = len(config_list)
#         num_tslices = int(len(data) / (num_sources*num_config))
#         data = np.reshape(data, (num_sources, num_config, num_tslices))
#         return data
