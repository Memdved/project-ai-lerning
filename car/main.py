"""Запускаемый файл."""


from src.game import Game


def main() -> None:
    game = Game((1280, 720), 144, debug=False)
    game.run()


if __name__ == "__main__":
    main()
