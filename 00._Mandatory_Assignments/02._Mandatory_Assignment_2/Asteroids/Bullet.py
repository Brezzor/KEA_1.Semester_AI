import math, pygame
from Utils import display_width, display_height, bullet_speed

WHITE = (255,255,255)
RADIUS = 3

class Bullet:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.dir = direction
        self.life = 30

    def updateBullet(self):
        self.x += bullet_speed * math.cos(math.radians(self.dir))
        self.y += bullet_speed * math.sin(math.radians(self.dir))
        self.x %= display_width
        self.y %= display_height
        self.life -= 1

    def drawBullet(self, surface):
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), RADIUS)