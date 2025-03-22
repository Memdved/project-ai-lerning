# main.py
"""Запускаемый файл."""

import os
import json
from src.game import Game
import pygame as pg


def main() -> None:
    """Главная функция."""

    # Load available models (drivers)
    model_dir = "src/models"
    available_models = [
        f.split(".json")[0] for f in os.listdir(model_dir) if f.endswith(".json")
    ]

    # Show menu
    print("Выберите действие:")
    print("1. Запустить обучение новой модели")
    if available_models:
        print("2. Продолжить обучение последней модели")
        print("3. Показать возможности лучшего водителя")

    choice = input("Введите номер действия: ")

    if choice == "1":
        game = Game((1280, 720), 144, debug=False)
    elif choice == "2" and available_models:
        last_model = sorted(available_models)[-1]
        game = Game((1280, 720), 144, debug=False, load_model=last_model)
    elif choice == "3" and available_models:
        last_model = sorted(available_models)[-1]
        game = Game((1280, 720), 144, debug=False, show_best=last_model)
    else:
        print("Неверный выбор. Запуск обучения новой модели.")
        game = Game((1280, 720), 144, debug=False)

    game.run()


if __name__ == "__main__":
    main()
