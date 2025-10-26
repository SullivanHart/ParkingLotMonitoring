class Plate():
    def __init__( self, text, prob ):
        self.text = text
        self.prob = prob

    def getText( self ):
        return self.color

    def getProb( self, style ):
        return self.style

    def to_dict( self ):
        return {
            "text" : self.text,
            "prob" : self.prob
        }