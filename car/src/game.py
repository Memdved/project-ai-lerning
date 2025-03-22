
# src/game.py
import abc
import pygame as pg
import random
import os
import json
from pygame import init, quit
from src.car import Car
from src.track import Track
from src.driver import Driver
from copy import copy, deepcopy


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

    def __init__(
        self,
        size: tuple[int],
        fps: int,
        debug: bool = False,
        load_model: str = None,
        show_best: str = None,
    ) -> None:
        init()

        self.size: tuple[int] = size
        self.__fps: int = fps
        self.__debug: bool = debug
        self.__paused: bool = False

        self.__screen: pg.Surface = pg.display.set_mode(self.size)

        self.__clock: pg.time.Clock = pg.time.Clock()

        self.__game_run: bool = True
        self.__track: Track = Track()

        # Genetic Algorithm parameters
        self.population_size = 500
        self.mutation_rate = 0.01
        self.generation = 0
        self.best_driver = None
        self.model_name = "model"
        self.model_dir = "src/models"
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize population
        self.population = []
        if load_model:
            self.load_population(load_model)
        elif show_best:
            self.load_and_show_best(show_best)
        else:
            self.population = [
                Driver(self.__track, [150, 300], 400, 250, (60, 30))
                for _ in range(self.population_size)
            ]

        self.show_best_only = bool(show_best)

        # Arrow image (replace 'arrow.png' with your arrow image file)
        self.arrow_image = pg.image.load("src/arrow.png")  # Create arrow.png
        self.arrow_image = pg.transform.scale(self.arrow_image, (40, 40))

    def __del__(self) -> None:
        quit()

    def run(self) -> None:
        while self.__game_run:
            self.dt = self.__clock.tick(self.__fps) / 1000.0
            self.check_events()
            if not self.__paused:
                self.logic()
            self.show()
            pg.display.set_caption(
                f"fps -- {round(self.__clock.get_fps(), 1)} | Generation: {self.generation}"
            )

    def check_events(self) -> None:
        event: pg.event.Event
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.__game_run = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    self.__paused = not self.__paused

    def logic(self) -> None:
        # Update all drivers
        for driver in self.population:
            driver.update(self.dt)
            driver.check_collision(self.__track)

        # Check if all drivers are dead
        if all(driver.dead for driver in self.population):
            self.evolve_population()

    def evolve_population(self):
        """Выполняет эволюцию популяции с использованием генетического алгоритма."""
        self.generation += 1

        # Calculate fitness for each driver
        for driver in self.population:
            driver.calculate_fitness()

        # Sort the population by fitness (descending order)
        self.population.sort(key=lambda driver: driver.fitness, reverse=True)

        # Save the best driver of the generation
        if (
            self.best_driver is None
            or self.population[0].fitness > self.best_driver.fitness
        ):
            self.best_driver = deepcopy(
                self.population[0]
            )  # save a deepcopy of the best driver
            self.save_model(self.best_driver, f"{self.model_dir}/best_driver.json")

        # Create a new population through crossover and mutation
        new_population = [self.best_driver]  #  elitism: always keep the best
        while len(new_population) < self.population_size:
            parent1 = random.choice(
                self.population[: self.population_size // 2]
            )  # Select from the best half
            parent2 = random.choice(self.population[: self.population_size // 2])

            child = parent1.crossover(parent2)
            child.mutate(self.mutation_rate)
            new_population.append(child)

        # Replace the old population with the new population
        self.population = new_population

        # Reset the drivers for the next generation
        for driver in self.population:
            driver.reset()

    def show(self) -> None:
        self.__screen.fill((0, 0, 0))

        # Show the track
        self.__track.show(self.__screen, [])

        # Show all drivers (you might want to show only the best for performance)
        if not self.show_best_only:
            for driver in self.population:
                driver.show(self.__screen, self.__debug)
        elif self.best_driver:
            self.best_driver.show(self.__screen, self.__debug)

        # Show arrow above the best driver
        if self.best_driver and not self.show_best_only:
            arrow_x = self.best_driver.car.get_center()[0] - self.arrow_image.get_width() / 2
            arrow_y = self.best_driver.car.get_center()[1] - 50  # Adjust position above the car
            self.__screen.blit(self.arrow_image, (arrow_x, arrow_y))
        elif self.best_driver and self.show_best_only:
            arrow_x = self.best_driver.car.get_center()[0] - self.arrow_image.get_width() / 2
            arrow_y = self.best_driver.car.get_center()[1] - 50  # Adjust position above the car
            self.__screen.blit(self.arrow_image, (arrow_x, arrow_y))

        pg.display.flip()

    def restart(self) -> None:
        self.__track: Track = Track()

    def save_model(self, driver, filename):
        """Сохраняет хромосому водителя в файл."""
        chromosome_data = [
            {"action": action, "duration": duration}
            for action, duration in driver.chromosome
        ]
        model_data = {
            "chromosome": chromosome_data,
            "max_velocity": driver.car.max_velocity,
            "acceleration": driver.car.acceleration,
            "size": driver.car.size,
        }

        with open(filename, "w") as f:
            json.dump(model_data, f, indent=4)
        print(f"Модель сохранена в {filename}")

    def load_population(self, model_name):
        """Загружает популяцию из файла."""
        filename = f"{self.model_dir}/{model_name}.json"
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден. Создание новой популяции.")
            self.population = [
                Driver(self.__track, [150, 300], 400, 250, (60, 30))
                for _ in range(self.population_size)
            ]
            return

        with open(filename, "r") as f:
            model_data = json.load(f)
        chromosome_data = model_data["chromosome"]
        chromosome = [
            (item["action"], item["duration"]) for item in chromosome_data
        ]  # convert dict to tuple

        self.best_driver = Driver(
            self.__track,
            [150, 300],
            model_data["max_velocity"],
            model_data["acceleration"],
            tuple(model_data["size"]),
            chromosome,
        )  # convert list to tuple
        self.population = [
            deepcopy(self.best_driver) for _ in range(self.population_size)
        ]  # create new population
        for driver in self.population:
            driver.reset()

        print(f"Модель загружена из {filename}")

    def load_and_show_best(self, model_name):
        """Загружает лучшего водителя из файла и отображает только его."""
        filename = f"{self.model_dir}/{model_name}.json"
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден.")
            return

        with open(filename, "r") as f:
            model_data = json.load(f)
        chromosome_data = model_data["chromosome"]
        chromosome = [
            (item["action"], item["duration"]) for item in chromosome_data
        ]  # convert dict to tuple

        self.best_driver = Driver(
            self.__track,
            [150, 300],
            model_data["max_velocity"],
            model_data["acceleration"],
            tuple(model_data["size"]),
            chromosome,
        )  # convert list to tuple
        self.population = [
            deepcopy(self.best_driver) for _ in range(self.population_size)
        ]  # create new population
        for driver in self.population:
            driver.reset()
        print(f"Модель загружена из {filename}")
        self.show_best_only = True
