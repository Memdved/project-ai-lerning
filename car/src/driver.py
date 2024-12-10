import numpy as np
import random
import os
import json
from src.track import get_lines_from_track_json
import pygame as pg

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.99, min_exploration_rate=0.1): #Увеличен min_exploration_rate
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}  # Используем словарь как Q-таблицу

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_size)  # Случайное действие (исследование)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
            return np.argmax(q_values)  # Действие с максимальным Q-значением (эксплуатация)

    def learn(self, state, action, reward, next_state):
        old_q_value = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q_value)
        self.q_table[(tuple(state), action)] = new_q_value

        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)


class Driver:
    def __init__(self, car, track, rays, reward_lines, model_path="q_learning_model.json", learning_rate=0.1, discount_factor=0.95):
        self.car = car
        self.track = track
        self.rays = rays
        self.reward_lines = reward_lines
        self.state_size = 4  # (x, y, angle, velocity)
        self.action_size = 4  # w, a, s, d
        self.agent = QLearningAgent(self.state_size, self.action_size, learning_rate=learning_rate, discount_factor=discount_factor)
        self.reset() # Добавлено
        self.reward_collected = [False] * len(self.reward_lines)
        self.model_path = model_path
        self.load_model()
        
    def get_state(self):
        cx, cy = self.car.get_center()
        return np.array([cx, cy, self.car.angle, self.car.velocity])

    def act(self, state):
        return self.agent.choose_action(state)

    def update(self, dt):
        state = self.get_state() # Добавлено получение состояния
        action = self.act(state)
        reward = self.get_reward(state)
        self.car.update({pg.K_w: action == 0, pg.K_s: action == 2, pg.K_a: action == 1, pg.K_d: action == 3}, dt)
        next_state = self.get_state()
        done = self.track.check_intersection(self.car.lines)
        if done:
            reward -= 100
        self.agent.learn(state, action, reward, next_state)
        self.save_model()

    def get_reward(self, state):
        cx, cy, angle, velocity = state
        reward = 0

        # Вознаграждение за прохождение контрольных точек
        for i, line in enumerate(self.reward_lines):
            if not self.reward_collected[i] and self._check_intersection_line(cx, cy, line):
                reward += 100
                self.reward_collected[i] = True

        # Штраф за столкновение со стеной
        min_distance_to_wall = min(self._distance_to_walls(cx, cy))
        if min_distance_to_wall < 30:
            reward -= (30 - min_distance_to_wall) * 5 # Сила штрафа

        # Бонус за скорость (не слишком большую)
        reward += velocity * 0.1
        if velocity > 5:
          reward -= (velocity -5) *0.5


        return reward

    def _distance_to_walls(self, x, y):
        distances = []
        for line in self.track.lines:
            distances.append(self._point_line_distance((x,y), line))
        return distances

    def _check_intersection_line(self, x, y, line):
        x1, y1 = line[0]
        x2, y2 = line[1]
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        dist_to_line = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return min_x <= x <= max_x and min_y <= y <= max_y and dist_to_line < 20

    def _point_line_distance(self, point, line):
        x0, y0 = point
        x1, y1 = line[0]
        x2, y2 = line[1]
        return abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def reset(self):
        self.car.restart()
        self.reward_collected = [False] * len(self.reward_lines)
        self.agent.q_table = {} # Переинициализация Q-таблицы

    def save_model(self):
        """Сохраняет Q-таблицу в JSON-файл."""
        serializable_q_table = {}
        for key, value in self.agent.q_table.items():
            serializable_key = str(key) # Преобразуем кортеж в строку
            serializable_q_table[serializable_key] = value

        with open(self.model_path, 'w') as f:
            json.dump(serializable_q_table, f, indent=4)

    def load_model(self):
        """Загружает Q-таблицу из JSON-файла."""
        try:
            with open(self.model_path, 'r') as f:
                loaded_q_table = json.load(f)
                self.agent.q_table = {}  #Очищаем старую таблицу
                for key, value in loaded_q_table.items():
                    deserialized_key = eval(key) # Преобразуем строку обратно в кортеж (ОСТОРОЖНО!)
                    self.agent.q_table[deserialized_key] = value

        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Файл модели '{self.model_path}' не найден или поврежден. Создается новая модель.")
