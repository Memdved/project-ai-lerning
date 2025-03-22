
# src/car.py
import pygame as pg
import abc
import math
import random


class BaseCar(abc.ABC):
    """Абстрактный класс, для описания работы машины."""

    @abc.abstractmethod
    def __init__(
        self, pos: list[int], max_velocity: float, acceleration: float, size: tuple
    ) -> None:
        pass

    @abc.abstractmethod
    def update(self, keys: tuple, dt: float) -> None:
        """Метод для обновления состояния машины."""

    @abc.abstractmethod
    def show(self, screen: pg.Surface, debug: bool):
        """Метод для отображения машины на экран.

        :param screen: Окно, куда будет отображаться машина.
        :type screen: pg.Surface
        """


class Car(BaseCar):
    """Класс для работы с машиной."""

    def __init__(
        self, pos: list[int], max_velocity: float, acceleration: float, size: tuple
    ) -> None:
        self.pos: list[int] = pos
        self.max_velocity: float = max_velocity
        self.acceleration: float = acceleration
        self.size: tuple = size
        self.angle: float = -90
        self.velocity: float = 0
        self.rotation_speed_factor = 0.15 * self.size[0] / self.size[1]
        self.update_lines()
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update_lines(self):
        self.lines: list = [
            [  # Левая стенка
                (self.pos[0], self.pos[1]),
                (
                    self.pos[0] + self.size[0] * math.cos(math.radians(self.angle)),
                    self.pos[1] + self.size[0] * math.sin(math.radians(self.angle)),
                ),
            ],
            [  # Нижняя стенка
                (self.pos[0], self.pos[1]),
                (
                    self.pos[0]
                    + self.size[1] * math.cos(math.radians(self.angle + 90)),
                    self.pos[1]
                    + self.size[1] * math.sin(math.radians(self.angle + 90)),
                ),
            ],
            [  # Верхняя стенка
                (
                    self.pos[0] + self.size[0] * math.cos(math.radians(self.angle)),
                    self.pos[1] + self.size[0] * math.sin(math.radians(self.angle)),
                ),
                (
                    self.pos[0]
                    + self.size[0] * math.cos(math.radians(self.angle))
                    + self.size[1] * math.cos(math.radians(self.angle + 90)),
                    self.pos[1]
                    + self.size[0] * math.sin(math.radians(self.angle))
                    + self.size[1] * math.sin(math.radians(self.angle + 90)),
                ),
            ],
            [  # Правая стенка
                (
                    self.pos[0]
                    + self.size[1] * math.cos(math.radians(self.angle + 90)),
                    self.pos[1]
                    + self.size[1] * math.sin(math.radians(self.angle + 90)),
                ),
                (
                    self.pos[0]
                    + self.size[0] * math.cos(math.radians(self.angle))
                    + self.size[1] * math.cos(math.radians(self.angle + 90)),
                    self.pos[1]
                    + self.size[0] * math.sin(math.radians(self.angle))
                    + self.size[1] * math.sin(math.radians(self.angle + 90)),
                ),
            ],
        ]

    def move(self, velocity_change: float, dt: float) -> None:
        self.velocity += self.acceleration * velocity_change * dt
        self.velocity = max(-self.max_velocity, min(self.velocity, self.max_velocity))

        dx = self.velocity * math.cos(math.radians(self.angle)) * dt
        dy = self.velocity * math.sin(math.radians(self.angle)) * dt
        self.pos[0] += dx
        self.pos[1] += dy
        self.update_lines()

    def update(self, keys: tuple, dt: float) -> None:
        w_pressed, s_pressed, a_pressed, d_pressed = keys #unpack tuple

        if w_pressed:
            self.move(1, dt)
        elif s_pressed:
            self.move(-1, dt)
        else:
            if self.velocity > 0:
                self.move(-1, dt)
            elif self.velocity < 0:
                self.move(1, dt)

        if a_pressed:
            rotation_speed = abs(self.velocity) * self.rotation_speed_factor
            self.angle -= rotation_speed * dt
            self.update_lines()
        if d_pressed:
            rotation_speed = abs(self.velocity) * self.rotation_speed_factor
            self.angle += rotation_speed * dt
            self.update_lines()
            

    def get_center(self) -> tuple:
        """Метод для получения центра машины."""
        return (self.pos[0] + self.size[0]/2, self.pos[1] + self.size[1]/2)

    def show(self, screen: pg.Surface, debug: bool) -> None:
        line: list
        for line in self.lines:
            pg.draw.line(screen, self.color, line[0], line[1], width=5)


    def __lines_intersect(
        self,
        line1: tuple[tuple[int, int], tuple[int, int]],
        line2: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[int, int] | None:
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if den == 0:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return int(x), int(y)
