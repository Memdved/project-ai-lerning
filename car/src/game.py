import abc
import pygame as pg
from pygame import init, quit
from src.car import Car
from src.track import Track
from src.ray import Rays

class BaseGame(abc.ABC):
    """Абстрактный класс, описывающий работу игры."""

    @abc.abstractmethod
    def __init__(self, size: list[int], fps: int, debug: bool = False) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> None:
        """Метод для запуска игры."""

    @abc.abstractmethod
    def check_events(self) -> None:
        """Метод для проверки ивентов."""

    @abc.abstractmethod
    def show(self) -> None:
        """Метод для отображения объектов на экран."""

    @abc.abstractmethod
    def logic(self) -> None:
        """Метод для работы с логикой игры."""
    
    @abc.abstractmethod
    def restart(self) -> None:
        """Метод для перезапуска игры."""


class Game(BaseGame):
    """Класс для работы с игрой."""

    def __init__(self, size: tuple[int], fps: int, debug: bool = False) -> None:
        init()

        self.size: tuple[int] = size
        self.__fps: int = fps
        self.__debug: bool = debug

        self.__screen: pg.Surface = pg.display.set_mode(self.size)

        self.__clock: pg.time.Clock = pg.time.Clock()

        self.__game_run: bool = True
        self.__car: Car = Car([150, 300], 400, 250, (60, 30))
        self.__track: Track = Track()
        self.__rays: Rays = Rays(100, self.__car.get_center(), 1000)
    
    

    def __del__(self) -> None:
        quit()

    def run(self) -> None:
        while self.__game_run:
            self.dt = self.__clock.tick(self.__fps) / 1000.0
            self.check_events()
            self.logic()
            self.show()
            pg.display.set_caption(f"fps -- {round(self.__clock.get_fps(), 1)}")

    def check_events(self) -> None:
        event: pg.event.Event
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.__game_run = False

    def logic(self) -> None:
        keys = pg.key.get_pressed()
        self.__car.update(keys, self.dt)
        self.__rays.update(self.__car.get_center(), 1000)
        if self.__track.check_intersection(self.__car.lines):
            self.restart()

    def show(self) -> None:
        self.__screen.fill((0, 0, 0))

        self.__car.show(self.__screen, self.__debug)
        self.__track.show(self.__screen, self.__car.lines)
        self.__rays.show(self.__screen, self.__debug, self.__track.lines)

        pg.display.flip()
    
    def restart(self) -> None:
        self.__game_run: bool = True
        self.__car: Car = Car([150, 300], 400, 250, (60, 30))
        self.__track: Track = Track()
        self.__rays: Rays = Rays(100, self.__car.get_center(), 1000)
