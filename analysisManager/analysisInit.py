# I need to initialize an analysis
# I need to know:
# - dealing with 'gauge' files or 'stats' (is this a problem for the reader? I don't think so, 
#   the reader can just take proper outerClass based on the file extension and assign the 'gauge' or 'stats' according to the analysisInitialization)
#   ? but is this going to undermine the flexibility? I will have to stick with either 'gauge' or 'stats' data.. this is not going to be feasible if I
#     have to store data from fits or similar (those are going to be always 'stats' file) 
# - choosing which statistics to use, either 'jack' or 'boot' and set their paramters, if any
# - (choosing the file format we are going to work with? )


class analysis:
    """Initializer for the analysis."""

    def __init__(self, *, stats_type, stats_par, tsrc_list, config_list): # I need also config list in principle
        self.tsrc_list = tsrc_list
        self.config_list = config_list
        self.stats_type = stats_type
        self.stats_par = stats_par
        
        # self.stats_fun = get_stats_fun(stats_type, stats_par)
        # self.err_fun = get_err_fun(stats_type)
    
    def dataStats(self, file):
        statsID = get_statsID(file)
        if statsID=='gauge':
            data = quick_gauge_h5reader(file, self.stats_type, self.stats_par, self.tsrc_list)
        elif statsID=='stats':
            data = quick_stats_h5reader(file, self.stats_type, self.stats_par)
        return data_stats(data, self.stats_type, self.stats_par)

    def corrStats(self, file):
        statsID = get_statsID(file)
        if statsID=='gauge':
            data = quick_gauge_h5reader(file, self.stats_type, self.stats_par, self.tsrc_list)
        elif statsID=='stats':
            data = quick_stats_h5reader(file, self.stats_type, self.stats_par)
        return corr(data, self.stats_type, self.stats_par)