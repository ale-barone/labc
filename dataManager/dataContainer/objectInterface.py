from abc import ABC, abstractmethod

# I may want to inlude here a shared structure for the different formats

class IReader(ABC):
    """Common interface for a reader. It defines the property/methods that every reader must have by definition."""
    
    def __init__(self, file):
        self.file = file 
        
    @abstractmethod
    def read(self):
        """Basic reading operation of a specific section."""
    
    @abstractmethod
    def dump(self):
        """Dump all the content of the file."""
        # maybe a dictionary?#
    
    @abstractmethod
    def get_structure(self):
        """Deploy whatever sub structure the file has."""


class IWriter(ABC):
    """Common interface for a writer. It defines the property/methods that every writer must have by definition."""
    
    def __init__(self, file, statsID):
        self.file = file 
        self.statsID = statsID 
        self._create_file()

    @abstractmethod
    def _create_file(self):
        """Creator for the file."""
        
    @abstractmethod
    def write(self, path, data):
        """Basic writing operation of data into location path."""

