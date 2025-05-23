import pygame
import random
from Player import Player
from Asteroid import Asteroid
from Bullet import Bullet
from Utils import display_width, display_height, black, white, player_size, isColliding

pygame.init()
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Asteroids")
timer = pygame.time.Clock()

def drawText(msg, color, x, y, s, center=True):
    screen_text = pygame.font.SysFont("Calibri", s).render(msg, True, color)
    if center:
        rect = screen_text.get_rect()
        rect.center = (x, y)
    else:
        rect = (x, y)
    gameDisplay.blit(screen_text, rect)

def gameLoop(startingState="Menu"):
    gameState = startingState
    bullet_capacity = 4
    bullets = []
    asteroids = []
    score = 0
    player = Player(display_width / 2, display_height / 2)

    # Spawn initial asteroids
    for _ in range(3):
        asteroids.append(Asteroid(random.randint(0, display_width), random.randint(0, display_height), "Large"))

    SPAWN_DELAY = 2000
    MAX_ASTEROIDS = 8
    next_spawn = pygame.time.get_ticks() + SPAWN_DELAY

    # Main GameLoop - 
    while gameState != "Exit":

        if gameState == "Menu":
            gameDisplay.fill(black)
            drawText("ASTEROIDS", white, display_width / 2, display_height / 2, 100)
            drawText("Press any key to START", white, display_width / 2, display_height / 2 + 100, 50)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gameState = "Exit"
                if event.type == pygame.KEYDOWN:
                    gameState = "Playing"
            continue

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameState = "Exit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.thrust = True
                if event.key == pygame.K_LEFT:
                    player.rtspd = -5
                if event.key == pygame.K_RIGHT:
                    player.rtspd = 5
                if event.key == pygame.K_SPACE and len(bullets) < bullet_capacity:
                    bullets.append(Bullet(player.x, player.y, player.dir))
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    player.thrust = False
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    player.rtspd = 0

        player.updatePlayer()
        for a in asteroids:
            a.updateAsteroid()
        for b in bullets:
            b.updateBullet()
            if  b.life <= 0:
                bullets.remove(b)

        # Bullet collisions
        for b in bullets[:]:
            for a in asteroids[:]:
                if isColliding(b.x, b.y, a.x, a.y, a.size):
                    bullets.remove(b)
                    asteroids.remove(a)
                    score += 10
                    break
        
        now = pygame.time.get_ticks()
        if now >= next_spawn and len(asteroids) < MAX_ASTEROIDS:
            x, y = random.choice([(0, random.randint(0, display_height)),
                                  (display_width, random.randint(0, display_height)),
                                  (random.randint(0, display_width), 0),
                                  (random.randint(0, display_width), display_height)])
            new_type = random.choices(["Large", "Normal", "Small"],
                                    weights=[3,2,1])[0]
            asteroids.append(Asteroid(x,y, new_type))
            next_spawn = now + SPAWN_DELAY

        gameDisplay.fill(black)

        for a in asteroids:          # <-- Tegn asteroider
            a.drawAsteroid(gameDisplay)

        for b in bullets:            # <-- Tegn bullets
            b.drawBullet(gameDisplay)

        player.drawPlayer(gameDisplay)
        drawText(f"Score: {score}", white, 60, 20, 40, False)
        pygame.display.update()
        timer.tick(30)

if __name__ == "__main__":
    gameLoop()
    pygame.quit()
    quit()