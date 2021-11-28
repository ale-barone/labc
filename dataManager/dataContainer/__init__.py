import numpy as np
from .HDF5.HDF5Reader import HDF5Reader
from .HDF5.HDF5Writer import HDF5Writer
from .HDF5.HDF5Formatter import HDF5Formatter
from .HDF5.HDF5Utilities import check_statsID, get_statsID
from .objectFactory import objectFactory

# I THINK IT WOULD MAKE MORE SENSE TO HAVE THIS ALL FILE INTO THE __init__.py of 'dataManager'

# I may have to build a fileStatsID method which takes its own reader/writer

# class fileStatsID:
    
#     def __init__(self, file, statsID):
#         self.file = file
#         self.statsID = statsID
#         self.reader = reader(file)
#         self.writer = writer(file, statsID)

reader_factory = objectFactory()
reader_factory.add_obj('.h5', HDF5Reader)

# final object to be called: it deals with files, whereas readerFactory deals directly only with readers (doesn't really know anything about files)
class reader:
    """
    Class which take as input the file, read its 'statsID' ('gauge' or 'stats', 'generic' if None) and calls the appropriate
    reader with the help of a reader factory.
    """

    def __new__(cls,  file): # TODO: in principle I should get the file ID from the attribute! (if there is no attribute then I go with 'generic')
        statsID = get_statsID(file)
        reader = reader_factory.get_obj(file, statsID)
        cls.reader = reader
        return reader(file)

    # this need some extra thinkg
    # @classmethod
    # def add_reader(cls, ext, reader):
    #     cls.factory.add_reader(ext, reader)
    #     #return __class__.factory


writer_factory = objectFactory()
writer_factory.add_obj('.h5', HDF5Writer)

class writer:
    """
    Class which take as input the file and its 'ID' (typically 'generic', 'gauge' or 'stats') and calls the appropriate
    writer with the help of a writer factory.
    """

    def __new__(cls, file, statsID): # TODO: in principle I should get the file ID from the attribute! (if there is no attribute then I go with 'generic')
        writer = writer_factory.get_obj(file, statsID)
        cls.writer = writer
        return writer(file, statsID)

formatter_factory = objectFactory()
formatter_factory.add_obj('.h5', HDF5Formatter)

class formatter:
    """
    Class which take an input file with 'statsID'={'gauge', 'stats'}
    and format the data in a numpy array 'data' such that
    data[0]: mean
    data[1]: err
    data[2:]: bins
    """

    def __new__(cls, file, statsType):
        check_statsID(file)
        statsID = get_statsID(file)
        formatter = formatter_factory.get_obj(file, statsID)
        cls.reader = formatter
        return formatter(file, statsType)



# TODO: write this thing more in general! This is NOT acceptable!
class extractor:
    
    def __init__(self, file_fun, *args):
        def _file_fun(tsrc, config):
            return file_fun(tsrc, config, *args)
        self.file_fun = _file_fun

    # not efficient, I call the reader every time in extract_all!
    def extract(self, tsrc, config, path, re):
        ext_reader = reader(self.file_fun(tsrc, config))
        data = np.roll(np.array(ext_reader.read(path, re=re)), -tsrc)  
        return data          

    
    def extract_all(self, tsrc_list, config_list, path, re):
        data = np.array([])
        for tsrc in tsrc_list:
            for config in config_list:
                data_single = self.extract(tsrc, config, path, re)
                data = np.append(data, data_single)
                
        num_sources = len(tsrc_list)
        num_config = len(config_list)
        num_tslices = int(len(data) / (num_sources*num_config))
        data = np.reshape(data, (num_sources, num_config, num_tslices))
        return data