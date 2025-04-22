import numpy as np
import random
import pygame
import pickle
from ga_models.ga_simple import SimpleModel
from Asteroids.Asteroids import Player, Asteroid, Bullet, Saucer, isColliding, display_width, display_height


class AsteroidsEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player = Player(display_width / 2, display_height / 2)
        self.asteroids = [Asteroid(random.randint(0, display_width), random.randint(0, display_height), "Large") for _ in range(3)]
        self.bullets = []
        self.score = 0
        self.steps = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        nearest = self._get_nearest_asteroid()
        dx, dy = 0, 0
        if nearest:
            dx = (nearest.x - self.player.x) / display_width
            dy = (nearest.y - self.player.y) / display_height

        speed = np.sqrt(self.player.hspeed**2 + self.player.vspeed**2) / 20  # normalize by max speed
        dir_norm = (self.player.dir % 360) / 360.0

        return np.array([
            speed,
            self.player.hspeed / 20,
            self.player.vspeed / 20,
            dx,
            dy,
            dir_norm
        ])


    def _get_nearest_asteroid(self):
        if not self.asteroids:
            return None
        return min(self.asteroids, key=lambda a: (a.x - self.player.x)**2 + (a.y - self.player.y)**2)

    def step(self, action):
        if self.done:
            return self._get_obs(), self.score, self.done

        # Decode action
        thrust, rotate_left, rotate_right, fire = action
        self.player.thrust = bool(thrust)
        self.player.rtspd = -5 if rotate_left else 5 if rotate_right else 0

        if fire and len(self.bullets) < 4:
            self.bullets.append(Bullet(self.player.x, self.player.y, self.player.dir))

        # Update state
        self.player.updatePlayer()
        for a in self.asteroids:
            a.updateAsteroid()
        for b in self.bullets:
            b.updateBullet()

        # Bullet collisions
        for b in self.bullets[:]:
            for a in self.asteroids[:]:
                if isColliding(b.x, b.y, a.x, a.y, a.size):
                    self.bullets.remove(b)
                    self.asteroids.remove(a)
                    self.score += 10
                    break

        # Check collision
        for a in self.asteroids:
            if isColliding(self.player.x, self.player.y, a.x, a.y, a.size):
                self.done = True

        self.steps += 1
        if self.steps > 1000:
            self.done = True

        return self._get_obs(), self.score, self.done


def evaluate_model(model, episodes=1):
    total_score = 0
    env = AsteroidsEnv()
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            output = model.update(obs)
            action = [1 if v > 0.5 else 0 for v in output[:4]]
            obs, reward, done = env.step(action)
        total_score += reward
    return total_score / episodes


def evolve_population(pop_size=20, generations=10):
    population = [SimpleModel(dims=(6, 8, 4)) for _ in range(pop_size)]
    for gen in range(generations):
        print(f"\nGeneration {gen+1}")
        scores = [(evaluate_model(model), model) for model in population]
        scores.sort(reverse=True, key=lambda x: x[0])
        print(f"Best score: {scores[0][0]}")

        next_gen = [scores[0][1], scores[1][1]]  # Keep top 2
        while len(next_gen) < pop_size:
            p1, p2 = random.choices(scores[:10], k=2)
            child = p1[1] + p2[1]
            child.mutate(0.1)
            next_gen.append(child)
        population = next_gen

    return scores[0][1]  # Return best model


def save_model(model, filename="00._Mandatory_Assignments/02._Mandatory_Assignment_2/models/best_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename="00._Mandatory_Assignments/02._Mandatory_Assignment_2/models/best_model.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    best = evolve_population()
    print("Training complete. Saving best model...")
    save_model(best)
    print("Model saved as best_model.pkl")

