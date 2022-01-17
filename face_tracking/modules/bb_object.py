import numpy as np

class BBox(object):
    def __init__(self, startX, startY, endX, endY, confidence):
        self.confidence = confidence
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY
        self.width = self.endX -self.startX
        self.height = self.endY -self.startY
        self.match = False

    def getWidth(self):

        return self.width
    def getHeight(self):

        return self.height
    def getArea(self):
        return self.width *self.height
    def getRec(self):
        return np.array([self.startX ,self.startY ,self.width ,self.height])
    def setMatch(self):
        self.match = True


