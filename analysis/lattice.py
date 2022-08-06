

RBC = {
    'C1': {
        'a': 0.011, # fm
        'ainv': 1.79, # GeV
        'L': 24,
        'T': 64,
    }
}

class Lattice:

    def __init__(self, ensemble):
        self.ensemble = ensemble
        self._ensemble_dict = RBC[ensemble]
        for key, value in self._ensemble_dict.items():
            setattr(self, key, value)
    
    def __str__(self):
        out = 30*"#" + "\n"
        out += "# Lattice\n"
        out += 30*"#" + "\n"
        out += f"# Ensemble '{self.ensemble}'\n"
        for key, value in self._ensemble_dict.items():
            out += f"#  {chr(183)}{key}: {value}\n"
        out += "############"
        return out
