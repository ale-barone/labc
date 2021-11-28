import h5py
from .HDF5Utilities import *
from ..objectInterface import IWriter


class AlreadyInitError(Exception):
    """Strucutre of the file has already been initiliazed!"""
    pass


class _HDF5WriterBase(IWriter):
    def __init__(self, file, statsID):
        super().__init__(file, statsID)  
    
    def _create_file(self):
        check_statsID(self.file) # to avoid writing file I did not create myself
        with h5py.File(self.file, 'w') as hf:
            hf.attrs['statsID'] = self.statsID
        self.groupStructure = False

    def init_groupStructure(self, groups):
        """Initializer for group structure of HDF5 file."""
        with h5py.File(self.file, 'r+') as hf:
            for group in list(groups):
                hf.create_group(group)
        self.groups = get_groups(self.file)
        self.paths = get_paths(self.file)


    
    def write(self, *args, **kwargs):
        """Wrapper around standard h5py.create_dataset method."""
        with h5py.File(self.file, 'r+') as hf:
            hf.create_dataset(*args, **kwargs)
    
    def set_attr(self, path, attr):
        # TODO: put distinctions between group/dataset?
        assert(path in get_paths(self.file)), "group/dataset is not in the tree"
        key, value = attr
        with h5py.File(self.file, 'r+') as hf:
            G = hf[path]
            G.attrs[key] = value


class HDF5Writer:
    class generic(_HDF5WriterBase):
        """Class for writing generic HDF5 file."""
        pass

    class gauge(_HDF5WriterBase):
        # I should probably put a global variable to deal with the general structure I want to impose

        def init_groupStructure(self, tsrc_list):
            """Create empty group structure of the form /corr/gauge{t_0{}, t_1{}, ...}."""
            # writer with structure tsrc{ config{dsets} }
            # create empty group structure
            if not self.groupStructure:
                self.groupStructure = True
                with h5py.File(self.file, 'r+') as hf:
                    G = hf.create_group('/corr')   
                    for tsrc in tsrc_list:
                        G.create_group(f't{tsrc}')
                
                self.groups = get_groups(self.file)
                self.tsrc_list = tsrc_list
            else:
                raise AlreadyInitError("Groups structure already initialized.")

        # TODO: I should figure out how to best make use of hd5f to see if I can resuscitate this
        # def add_config(self, tsrc, config, data):
        #     """Add gauge configuration data"""
        #     # TODO: add a proper 'raise' error if self.groupStructure==false (some custom error maybe) 
        #     with h5py.File(self.file, 'r+') as hf:
        #         G_tsrc = hf[f'/corr/t{tsrc}']  
        #         G_tsrc.create_dataset(f'config{config}', data=data)#shape=data.shape, dtype=data.dtype, data=data)
        #     self.datasets = get_datasets(self.file)

        def add_tsrc(self, tsrc, data):
            """Add gauge configuration data"""
            # TODO: add a proper 'raise' error if self.groupStructure==false (some custom error maybe) 
            with h5py.File(self.file, 'r+') as hf:
                G_tsrc = hf[f'/corr/t{tsrc}']  
                G_tsrc.create_dataset(f't{tsrc}', data=data)#shape=data.shape, dtype=data.dtype, data=data)
            self.datasets = get_datasets(self.file)

                
   

    class stats(_HDF5WriterBase):

        def init_groupStructure(self, stats_type, stats_par):
            if not self.groupStructure:
                self.groupStructure = True
                with h5py.File(self.file, 'r+') as hf:
                    hf.create_group('mean')
                    hf.create_group('err')
                    hf.attrs['stats_type'] = stats_type
                    Gbins = hf.create_group('bins')
                    # TODO: nicer information here (must be compatible with checks in dataAnalysis)
                    if stats_type=='boot':
                        num_bins, seed = stats_par 
                        Gbins.attrs['stats_par'] = str(num_bins) + ', ' + str(seed)
                    elif stats_type=='jack':
                        num_bins = stats_par
                        Gbins.attrs['stats_par'] = str(num_bins) 
                    self.groups = get_groups(self.file)
            else:
                raise AlreadyInitError("Groups structure already initialized.")
        
        # SHOULD I LIVE MORE FREEDOM? LIKE HERE?
        # I think the freedom should be left only in the standard method write
        # AT MOST, I SHOULD CONTROL THIS A BIT BETTER MYSELF BY READING MORE DOCUMENTATION OF h5py
        # def add_mean(self, *args, **kwargs):
        #     with h5py.File(self.file, 'r+') as hf:
        #         G = hf['/mean']
        #         G.create_dataset('mean', *args, **kwargs)

        def add_mean(self, array_mean):
            with h5py.File(self.file, 'r+') as hf:
                G = hf['/mean']
                G.create_dataset('mean', data=array_mean)
        
        def add_err(self, array_err):
            with h5py.File(self.file, 'r+') as hf:
                G = hf['/err']
                G.create_dataset('err', data=array_err)

        def add_bins(self, array_bins):
            with h5py.File(self.file, 'r+') as hf:
                G = hf['/bins']
                G.create_dataset('bins', data=array_bins)
                

