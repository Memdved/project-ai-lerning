# src/driver.py
import random
import pygame as pg
from src.car import Car
import math
import copy

class Driver:
    """Класс для управления машиной с использованием генетического алгоритма."""

    def __init__(self, track, initial_pos, max_velocity, acceleration, size, chromosome=None):
        self.track = track
        self.car = Car(initial_pos, max_velocity, acceleration, size)
        self.initial_pos = initial_pos
        self.chromosome = chromosome if chromosome else self.create_random_chromosome()
        self.action_index = 0
        self.fitness = 0
        self.distance_traveled = 0
        self.time_alive = 0
        self.dead = False

    def create_random_chromosome(self, chromosome_length=200):
        """Создает случайную хромосому."""
        chromosome = []
        for _ in range(chromosome_length):
            action = random.choice(['forward', 'backward', 'left', 'right', 'right', 'forward',])
            duration = random.uniform(0.05, 0.2)  # Длительность действия
            chromosome.append((action, duration))
        return chromosome

    def update(self, dt):
        """Выполняет действие из хромосомы и обновляет состояние машины."""
        if self.dead:
            return

        if self.action_index < len(self.chromosome):
            action, duration = self.chromosome[self.action_index]
            self.perform_action(action, dt)
            self.time_alive += dt

            if self.time_alive >= duration:
                self.time_alive = 0
                self.action_index += 1
        else:
            self.dead = True  # No more actions left

    def perform_action(self, action, dt):
        """Выполняет действие над машиной."""
        keys = {
            'forward': pg.K_w,
            'backward': pg.K_s,
            'left': pg.K_a,
            'right': pg.K_d,
            'none': None
        }

        # Create a dictionary of booleans representing pressed keys
        pressed_keys = {key: False for key in keys.values() if key is not None}

        if action != 'none':
            pressed_keys[keys[action]] = True

        # Create a tuple of booleans in the correct order (W, S, A, D)
        key_tuple = (pressed_keys.get(pg.K_w, False),
                     pressed_keys.get(pg.K_s, False),
                     pressed_keys.get(pg.K_a, False),
                     pressed_keys.get(pg.K_d, False))

        self.car.update(key_tuple, dt)
        self.distance_traveled += self.car.velocity * dt # Обновляем пройденное расстояние
    def calculate_fitness(self):
        """Вычисляет фитнес на основе пройденного расстояния и времени."""
        #  + self.time_alive
        if self.dead:
             self.fitness = self.distance_traveled
        else:
            self.fitness =  self.distance_traveled / (len(self.chromosome) + 1)  # Добавим небольшую награду, если не умер

        return self.fitness


    def show(self, screen, debug=False):
        """Отображает машину на экране."""
        self.car.show(screen, debug)

    def check_collision(self, track):
        """Проверяет столкновение машины с трассой."""
        if track.check_intersection(self.car.lines):
            self.dead = True
            return True
        return False

    def reset(self):
        """Сбрасывает состояние машины."""
        self.car = Car(self.initial_pos, self.car.max_velocity, self.car.acceleration, self.car.size)
        self.action_index = 0
        self.fitness = 0
        self.distance_traveled = 0
        self.time_alive = 0
        self.dead = False
        self.car.pos = list(self.initial_pos)  # Reset position
        self.car.angle = -90 # Reset angle
        self.car.velocity = 0 # Reset velocity
        self.car.update_lines()


    def crossover(self, other):
        """Выполняет скрещивание с другим водителем."""
        crossover_point = random.randint(1, len(self.chromosome) - 1)
        child_chromosome = self.chromosome[:crossover_point] + other.chromosome[crossover_point:]
        return Driver(self.track, self.initial_pos, self.car.max_velocity, self.car.acceleration, self.car.size, chromosome=child_chromosome)

    def mutate(self, mutation_rate=0.01):
        """Выполняет мутацию хромосомы."""
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                action = random.choice(['forward', 'backward', 'left', 'right', 'right', 'forward',])
                duration = random.uniform(0.05, 0.2)
                self.chromosome[i] = (action, duration)
