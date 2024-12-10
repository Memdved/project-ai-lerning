"""Файл для работы с трэком."""


import pygame as pg
import json
import os
import abc


def get_lines_from_track_json(project_path, file):
    """
    Извлекает список линий из файла track.json.

    Args:
        project_path: Путь к корневой папке проекта.

    Returns:
        Список линий (список списков координат), или None, если файл не найден или произошла ошибка.
    """
    try:
        track_json_path = os.path.join(project_path, "src", "tracks", file)
        with open(track_json_path, 'r') as f:
            data = json.load(f)
            return data["lines"]
    except FileNotFoundError:
        print(f"Файл track.json не найден по пути: {track_json_path}")
        return None
    except KeyError:
        print(f"Ключ 'lines' отсутствует в файле track.json")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка разбора JSON файла: {track_json_path}")
        return None
    

class BaseTrack(abc.ABC):
    """Абстрактный класс, описывающий работу трэка."""
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def check_intersection(self, car_lines: list) -> bool:
        """Метод для определения того, пересекается ли машина с трэком."""
    
    @abc.abstractmethod
    def show(self, screen: pg.Surface) -> None:
        """Метод для показа трэка на экране.

        :param screen: Экран
        :type screen: pg.Surface
        """


class Track(BaseTrack):
    """Класс для работы с трэком."""
    def __init__(self) -> None:
        self.lines: list = get_lines_from_track_json(os.getcwd(), "track_1.json")
    
    def check_intersection(self, car_lines: list) -> bool:
        """Проверяет пересечение линий машины с линиями трека."""
        if not self.lines or not car_lines:
            return False

        for car_line in car_lines:
            for track_line in self.lines:
                if self._lines_intersect(car_line, track_line):
                    return True
        return False

    def _lines_intersect(self, line1, line2):
        """Проверяет пересечение двух отрезков."""
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if den == 0:
            return False

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

        return 0 <= ua <= 1 and 0 <= ub <= 1

    def show(self, screen: pg.Surface, car_lines: list) -> None:
        for line in self.lines:
            if self.check_intersection(car_lines):
                pg.draw.line(screen, (255, 0, 0), line[0], line[1], width=5)
            else:
                pg.draw.line(screen, (255, 255, 255), line[0], line[1], width=5)
