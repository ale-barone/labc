import h5py
from .utilities import check_fileID, get_datasets, get_paths, get_groups
from .._objectInterface import IWriter


class AlreadyInitError(Exception):
    """Strucutre of the file has already been initiliazed!"""
    pass


class _WriterBase(IWriter):
    def __init__(self, file, fileID):
        super().__init__(file, fileID)  
    
    # OVERLOAD INTERFACE
    def _create_file(self):
        check_fileID(self.file) # to avoid writing file I did not create myself
        with h5py.File(self.file, 'w') as hf:
            hf.attrs['fileID'] = self.fileID
    
    def write(self, group, dataset, data, *args, **kwargs):
        """Wrapper around standard h5py.create_dataset method."""
        with h5py.File(self.file, 'r+') as hf:
            G = hf[f"{group}"]  
            G.create_dataset(dataset, *args, data=data, **kwargs)#shape=data.shape, dtype=data.dtype, data=data)
            self.datasets = get_datasets(self.file)
    
    # NEW METHODS
    def add_group(self, *groups, track_order=None):
        """Add group(s) to the file."""
        with h5py.File(self.file, 'r+') as hf:
            for group in groups:
                hf.create_group(group, track_order=track_order)
    
    def add_dataset(self, *args, **kwargs):
        """Wrapper around standard h5py.create_dataset method. From the method
        'create_dataset' in h5py:

        ########################################################################
        Create a new HDF5 dataset

        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        shape
            Dataset shape.  Use "()" for scalar datasets.  Required if "data"
            isn't provided.
        dtype
            Numpy dtype or string.  If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
        data
            Provide data to initialize the dataset.  If used, you can omit
            shape and dtype arguments.

        Keyword-only arguments:

        chunks
            (Tuple) Chunk shape, or True to enable auto-chunking.
        maxshape
            (Tuple) Make the dataset resizable up to this shape.  Use None for
            axes you want to be unlimited.
        compression
            (String or int) Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc. If specifying a dynamically loaded compression filter
            number, this must be a tuple of values.
        scaleoffset
            (Integer) Enable scale/offset filter for (usually) lossy
            compression of integer or floating-point data. For integer
            data, the value of scaleoffset is the number of bits to
            retain (pass 0 to let HDF5 determine the minimum number of
            bits necessary for lossless compression). For floating point
            data, scaleoffset is the number of digits after the decimal
            place to retain; stored values thus have absolute error
            less than 0.5*10**(-scaleoffset).
        shuffle
            (T/F) Enable shuffle filter.
        fletcher32
            (T/F) Enable fletcher32 error detection. Not permitted in
            conjunction with the scale/offset filter.
        fillvalue
            (Scalar) Use this value for uninitialized parts of the dataset.
        track_times
            (T/F) Enable dataset creation timestamps.
        track_order
            (T/F) Track attribute creation order if True. If omitted use
            global default h5.get_config().track_order.
        external
            (List of tuples) Sets the external storage property, thus
            designating that the dataset will be stored in one or more
            non-HDF5 file(s) external to the HDF5 file. Adds each listed
            tuple of (file[, offset[, size]]) to the dataset's list of
            external files.
        ########################################################################
        """
        with h5py.File(self.file, 'r+') as hf:
            hf.create_dataset(*args, **kwargs)
    
    def set_attr(self, path, attr):
        # TODO: put distinctions between group/dataset?
        assert(path in get_paths(self.file)), "group/dataset is not in the tree"
        key, value = attr
        with h5py.File(self.file, 'r+') as hf:
            G = hf[path]
            G.attrs[key] = value


class Writer:
    class generic(_WriterBase):
        """Class for writing generic HDF5 file."""
        pass

    class gauge(_WriterBase):
        # I should probably put a global variable to deal with the general structure I want to impose
                 

        # def init_groupStructure(self, tsrc_list):
        #     """Create empty group structure of the form /corr/gauge{t_0{}, t_1{}, ...}."""
        #     # writer with structure tsrc{ config{dsets} }
        #     # create empty group structure
        #     if not self.groupStructure:
        #         self.groupStructure = True
        #         with h5py.File(self.file, 'r+') as hf:
        #             G = hf.create_group('/corr')   
        #             # for tsrc in tsrc_list:
        #             #     G.create_group(f't{tsrc}')
                
        #         self.groups = get_groups(self.file)
        #         self.tsrc_list = tsrc_list
        #     else:
        #         raise AlreadyInitError("Groups structure already initialized.")

        def add_tsrc(self, group, tsrc, data, *args, **kwargs):
            """Add gauge configuration data"""
            # TODO: add a proper 'raise' error if self.groupStructure==false (some custom error maybe) 
            with h5py.File(self.file, 'r+') as hf:
                G_tsrc = hf[f"{group}"]  
                G_tsrc.create_dataset(f'tsrc{tsrc}', *args, data=data, **kwargs)
            self.datasets = get_datasets(self.file)

                
   

    class stats(_WriterBase):

        # def init_groupStructure(self, statsType):
        #     if not self.groupStructure:
        #         self.groupStructure = True
        #         with h5py.File(self.file, 'r+') as hf:
        #             hf.create_group('mean')
        #             hf.create_group('err')
        #             hf.attrs['statsType'] = statsType.ID
        #             Gbins = hf.create_group('bins')
        #             # TODO: nicer information here (must be compatible with checks in dataAnalysis)
        #             if statsType.ID=='boot':
        #                 num_bins, seed = statsType.num_bins, statsType.seed 
        #                 Gbins.attrs['statsPar'] = str(num_bins) + ', ' + str(seed)
        #             elif statsType.ID=='jack':
        #                 num_bins = statsType.num_bins
        #                 Gbins.attrs['statsPar'] = str(num_bins) 
        #             self.groups = get_groups(self.file)
        #     else:
        #         raise AlreadyInitError("Groups structure already initialized.")

        def add_stats_group(self, statsType, *groups, track_order=None):
            """Add group(s) to the file."""
            with h5py.File(self.file, 'r+') as hf:
                for group in groups:
                    hf.create_group(group, track_order=track_order)
                    self.set_attr(group, ('StatsType', statsType.__repr__()))

        def add_mean(self, group, array_mean, *args, **kwargs):
            with h5py.File(self.file, 'r+') as hf:
                G = hf[group]
                G.create_dataset('mean', *args, data=array_mean, **kwargs)
        
        def add_err(self, group, array_err, *args, **kwargs):
            with h5py.File(self.file, 'r+') as hf:
                G = hf[group]
                G.create_dataset('err', *args, data=array_err, **kwargs)

        def add_bins(self, group, array_bins, *args, **kwargs):
            with h5py.File(self.file, 'r+') as hf:
                G = hf[group]
                G.create_dataset('bins', *args, data=array_bins, **kwargs)
        
        def add_dataStats(self, group, dataStats, *args, **kwargs):
            with h5py.File(self.file, 'r+') as hf:
                G = hf[group]
                G.create_dataset('mean', *args, data=dataStats.mean, **kwargs)
                G.create_dataset('err', *args, data=dataStats.err, **kwargs)
                G.create_dataset('bins', *args, data=dataStats.bins, **kwargs)

