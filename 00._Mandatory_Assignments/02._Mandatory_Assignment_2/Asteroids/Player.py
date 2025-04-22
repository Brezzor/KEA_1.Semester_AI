import math
import pygame
from Utils import display_width, display_height, player_size, fd_fric, bd_fric, white

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.hspeed = 0
        self.vspeed = 0
        self.dir = -90
        self.rtspd = 0
        self.thrust = False

    def updatePlayer(self):
        speed = math.sqrt(self.hspeed**2 + self.vspeed**2)
        if self.thrust:
            if speed + fd_fric < 20:
                self.hspeed += fd_fric * math.cos(math.radians(self.dir))
                self.vspeed += fd_fric * math.sin(math.radians(self.dir))
            else:
                self.hspeed = 20 * math.cos(math.radians(self.dir))
                self.vspeed = 20 * math.sin(math.radians(self.dir))
        else:
            self.hspeed *= 0.99
            self.vspeed *= 0.99
        self.x = (self.x + self.hspeed) % display_width
        self.y = (self.y + self.vspeed) % display_height
        self.dir += self.rtspd

    def drawPlayer(self, surface):
        a = math.radians(self.dir)
        x = self.x
        y = self.y
        s = player_size
        # Nose point
        nose = (x + s * math.cos(a), y + s * math.sin(a))
        left = (x - s * math.cos(a + math.pi / 4), y - s * math.sin(a + math.pi / 4))
        right = (x - s * math.cos(a - math.pi / 4), y - s * math.sin(a - math.pi / 4))

        pygame.draw.line(surface, white, left, nose)
        pygame.draw.line(surface, white, right, nose)
        pygame.draw.line(surface, white, left, right)

        if self.thrust:
            flame_left = (x - s * math.cos(a + math.pi / 3), y - s * math.sin(a + math.pi / 3))
            flame_right = (x - s * math.cos(a - math.pi / 3), y - s * math.sin(a - math.pi / 3))
            tail = (x - 1.5 * s * math.cos(a), y - 1.5 * s * math.sin(a))
            pygame.draw.polygon(surface, white, [flame_left, flame_right, tail])