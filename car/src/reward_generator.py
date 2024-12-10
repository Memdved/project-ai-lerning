import pygame
import json
import os
from abc import ABC, abstractmethod
from track import Track


class TrackGenerator(ABC):
    """
    Абстрактный базовый класс для генераторов полос наград.
    """

    @abstractmethod
    def generate_track(self):
        """
        Генерирует полосу награды и сохраняет ее.
        """
        pass


class PygameTrackGenerator(TrackGenerator):
    """
    Генератор наград, использующий Pygame для визуального редактирования.
    """

    def __init__(self, width=1280, height=720, track_dir="src/rewards"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Награда генератор")
        self.track_dir = os.path.join(os.getcwd(), track_dir)
        self.lines = []
        self.current_line = []
        self.line_started = False
        self.__track: Track = Track()

    def generate_track(self):
        """
        Генерирует награду с помощью Pygame и сохраняет его в файл.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event)

            self._draw_track()
            pygame.display.flip()

        self._save_track()
        pygame.quit()

    def _handle_mouse_click(self, event):
        """
        Обрабатывает нажатие кнопки мыши.
        """
        if not self.line_started:
            self.current_line = [event.pos]  # Initialize as a list
            self.line_started = True
        else:
            self.current_line.append(event.pos)
            self.lines.append(tuple(map(tuple, self.current_line)))
            self.current_line = []
            self.line_started = False

    def _draw_track(self):
        """
        Отрисовывает награды на экране.
        """
        self.screen.fill((255, 255, 255))

        for line in self.lines:
            if len(line) > 1:
                pygame.draw.line(self.screen, (0, 0, 0), line[0], line[1], 2)

        if self.line_started and len(self.current_line) > 0:
            pygame.draw.line(
                self.screen, (0, 0, 0), self.current_line[0], pygame.mouse.get_pos(), 2
            )

        for line_t in self.__track.lines:
            pygame.draw.line(self.screen, (255, 0, 0), line_t[0], line_t[1], width=5)

    def _save_track(self):
        """
        Сохраняет трек в JSON-файл.
        """
        if not os.path.exists(self.track_dir):
            os.makedirs(self.track_dir)

        track_index = 1
        while os.path.exists(
            os.path.join(self.track_dir, f"line_reward{track_index}.json")
        ):
            track_index += 1

        track_data = {"lines": self.lines}

        with open(
            os.path.join(self.track_dir, f"line_reward{track_index}.json"), "w"
        ) as f:
            json.dump(track_data, f, indent=4)

        print(
            f"Трек сохранен в файл line_reward{track_index}.json в папке {self.track_dir}"
        )


class TrackData:
    """
    Класс для хранения данных трека.
    """

    def __init__(self, lines: list[list[tuple[int, int]]]):
        self.lines = lines

    def to_dict(self):
        return {"lines": self.lines}


if __name__ == "__main__":
    generator = PygameTrackGenerator()
    generator.generate_track()

