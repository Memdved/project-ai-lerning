# src/track.py
import pygame as pg
import json
import os

class Track:
    """Класс для работы с трассой."""

    def __init__(self, track_file="src/tracks/track_6.json"):
        self.lines = self.load_track(track_file)

    def load_track(self, track_file):
        """Загружает трассу из JSON-файла."""
        if not os.path.exists(track_file):
            print(f"Файл {track_file} не найден. Создание пустой трассы.")
            return []

        with open(track_file, "r") as f:
            track_data = json.load(f)
        return track_data["lines"]

    def show(self, screen: pg.Surface, debug: bool):
        """Отображает трассу на экране."""
        for line in self.lines:
            pg.draw.line(screen, (255, 255, 255), line[0], line[1], width=2)

    def check_intersection(self, car_lines):
        """Проверяет пересечение линий машины и трассы."""
        for car_line in car_lines:
            for track_line in self.lines:
                if self.__lines_intersect(car_line, track_line):
                    return True
        return False

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
