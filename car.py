class Car():
    def __init__( self, color = None, style = None, plates = None ):
        self.color = color
        self.style = style
        self.plates = plates or []

    def setColor( self, color ):
        self.color = color

    def getColor( self ):
        return self.color

    def setStyle( self, style ):
        self.style = style

    def getStyle( self, style ):
        return self.style

    def addPlate( self, plate ):
        self.plates.append( plate )

    def getPlates( self ):
        return self.plates

    def to_dict( self ):
        return {
            "color" : self.color,
            "style" : self.style,
            "plates" : self.plates
        }