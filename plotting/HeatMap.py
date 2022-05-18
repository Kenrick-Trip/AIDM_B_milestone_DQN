class HeatMap:
    def __init__(self, env, uncertainty):
        self.env = env
        self.uncertainty = uncertainty

    def generate1D(self):
        print(self.uncertainty.get_visit_count([-0.5]))