from .statsJack import statsJack
from .statsBoot import statsBoot 
import enum

# class stats_type(enum):
#     pass 

class statsManager:
    def __init__():

class stats_type:

    class jack:
        def __new__(cls, *, num_config):
            return statsJack(num_config=num_config)
    
    class boot:
        def __new__(cls, *, num_config, num_bins, seed=0):
            return statsBoot(num_config=num_config, num_bins=num_bins, seed=seed)




