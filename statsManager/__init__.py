from .statsJack import statsJack
from .statsBoot import statsBoot 

class statsType:

    class jack:
        def __new__(cls, *, num_config):
            return statsJack(num_config)
    
    class boot:
        def __new__(cls, *, num_config, num_bins, seed=0):
            return statsBoot(num_config=num_config, num_bins=num_bins, seed=seed)




