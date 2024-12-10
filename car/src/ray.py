"""Модуль для работы с лучами."""

import abc
from math import sin, cos, radians
import pygame as pg
import math


class BaseRay(abc.ABC):
    """Абстрактный класс, описывающий работу луча."""

    @abc.abstractmethod
    def __init__(self, pos: list, angle: float, length: int) -> None:
        pass

    @abc.abstractmethod
    def update_line(self, pos: list, angle: float) -> None:
        """Метод для обновления луча."""

    @abc.abstractmethod
    def get_line(self) -> list[tuple[int, int]]:
        """Метод для возвращения координат линии."""

    @abc.abstractmethod
    def check_intersection(
        self, track_lines: list
    ) -> tuple[bool, tuple[int, int] | None]:
        """Проверяет пересечение луча с линиями трека."""

    @abc.abstractmethod
    def _lines_intersect(
        self,
        line1: tuple[tuple[int, int], tuple[int, int]],
        line2: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[int, int] | None:
        """Проверяет пересечение двух отрезков."""

    @abc.abstractmethod
    def show(self, screen: pg.Surface, debug: bool = False):
        """Показывает лучи на экране."""

    @abc.abstractmethod
    def get_point(self) -> tuple:
        """Метод для возвращения точки пересечения с трэком."""


class BaseRays(abc.ABC):
    """Абстрактный класс, описывающий работу лучей вместе."""

    @abc.abstractmethod
    def __init__(self, num: int) -> None:
        pass

    @abc.abstractmethod
    def update(self) -> None:
        """Метод для обновления лучей."""

    @abc.abstractmethod
    def generate_rays(self, num: int) -> None:
        """Метод для генерации лучей."""

    @abc.abstractmethod
    def show(self, screen) -> None:
        """Метод для отображения всех лучей"""
    
    @abc.abstractmethod
    def get_points(self) -> list:
        """Метод для получения точек пересечения с треком."""


class Ray(BaseRay):
    """Класс для работы с лучами."""

    def __init__(self, pos: list, angle: float, length: int) -> None:
        self.__pos: list = pos
        self.__angle: float = angle
        self.__length: int = length
        self.__line: list = []
        self.__point: list = ()
        self.update_line(self.__pos, self.__angle)

    def update_line(self, pos: list, angle: float) -> None:
        self.__pos = pos
        self.__angle = angle
        self.__line = [
            (self.__pos[0], self.__pos[1]),
            (
                int(self.__pos[0] + self.__length * cos(radians(angle))),
                int(self.__pos[1] + self.__length * sin(radians(angle))),
            ),
        ]

    def get_line(self) -> list:
        return self.__line

    def check_intersection(
        self, track_lines: list
    ) -> tuple[bool, tuple[int, int] | None]:
        if not self.__line or not track_lines:
            return False, None

        closest_intersection = None
        min_distance = float("inf")

        for track_line in track_lines:
            intersection_point = self._lines_intersect(self.__line, track_line)
            if intersection_point:
                distance = math.dist(self.__pos, intersection_point)
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = intersection_point

        return bool(closest_intersection), closest_intersection

    def _lines_intersect(
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
        else:
            return None
    
    def get_point(self) -> tuple:
        return self.__point

    def show(self, screen, track_lines: list, debug=False):
        is_intersection, pos = self.check_intersection(track_lines)
        if is_intersection:
            self.__point = pos
            if debug:
                pg.draw.line(screen, (0, 255, 0), self.__line[0], pos, 3)
            pg.draw.circle(screen, (0, 255, 0), self.__point, 6)
        else:
            if debug:
                pg.draw.line(screen, (0, 255, 0), self.__line[0], self.__line[1], 3)


class Rays(BaseRays):
    """Класс для объединения лучей."""

    def __init__(self, num: int, pos: list, length: int) -> None:
        self.__rays: list[Ray]
        self.__num: int = num
        self.__points: list = []
        self.generate_rays(num, pos, length)

    def update(self, pos: list, length: int) -> None:
        self.generate_rays(self.__num, pos, length)
        ray: Ray
        for ray in self.__rays:
            self.__points.append(ray.get_point())

    def generate_rays(self, num: int, pos: list, length: int) -> None:
        self.__rays = [Ray(pos, (360 / num) * i, length) for i in range(num)]

    def show(self, screen: pg.Surface, debug: bool, track_lines: list) -> None:
        ray: Ray
        for ray in self.__rays:
            ray.show(screen, track_lines, debug)

    def get_points(self) -> list:
        return self.__points
