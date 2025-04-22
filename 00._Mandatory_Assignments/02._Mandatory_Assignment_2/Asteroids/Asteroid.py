import random
import math
from Utils import display_width, display_height

class Asteroid:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t
        self.size = 30 if t == "Large" else 20 if t == "Normal" else 10
        self.speed = random.uniform(1, (40 - self.size) * 4 / 15)
        self.dir = math.radians(random.randint(0, 360))

    def updateAsteroid(self):
        self.x += self.speed * math.cos(self.dir)
        self.y += self.speed * math.sin(self.dir)
        self.x %= display_width
        self.y %= display_height
