from fuller.mrfRec import MrfRec

class MrfRecPara(MrfRec):
    """
    A derivative class of MrfRec which expands on it by introducing multi-band reconstructions
    """
    def __init__(self, numBands, E_0):
        super().__init__(self)
        
        # modify the E_0 argument to support multiple arguments
        self.numBands = 
