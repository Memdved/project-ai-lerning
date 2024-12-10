
import pygame
import random
import numpy as np
import tensorflow as tf
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, message="Argument input_shape is deprecated")

# --- Classes GameObject, Snake, and Food ---

class GameObject:
    def __init__(self, x, y, color, size):
        self.x = x
        self.y = y
        self.color = color
        self.size = size

    def draw(self, surface):
        raise NotImplementedError

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)


class Snake(GameObject):
    def __init__(self, x, y, color, size):
        super().__init__(x, y, color, size)
        self.body = [(x, y)]
        self.direction = (1, 0)
        self.head_color = self.darken_color(color)
        self.snake_length = 4
        self.game_over = False

    def darken_color(self, color):
        r, g, b = color
        return (max(0, r - 50), max(0, g - 50), max(0, b - 50))

    def draw(self, surface):
        pygame.draw.rect(surface, self.head_color, (self.body[0][0], self.body[0][1], self.size, self.size))
        for segment in self.body[1:]:
            pygame.draw.rect(surface, self.color, (segment[0], segment[1], self.size, self.size))

    def move(self, direction):
        head_x, head_y = self.body[0]
        new_head = (head_x + direction[0] * self.size, head_y + direction[1] * self.size)

        if new_head in self.body:
            self.game_over = True

        self.body.insert(0, new_head)

        if len(self.body) > self.snake_length:
            self.body.pop()

    def grow(self):
        self.snake_length += 1
        tail_x, tail_y = self.body[-1]
        new_tail = (tail_x, tail_y)
        self.body.append(new_tail)

    def check_collision(self):
        head = self.body[0]
        for segment in self.body[1:]:
            if head == segment:
                return True
        return False


class Food(GameObject):
    def __init__(self, x, y, color, size):
        super().__init__(x, y, color, size)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.size, self.size))


# --- Class Game ---

class Game:
    def __init__(self, width, height, cell_size):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.snake = Snake(width // 2, height // 2, (0, 255, 0), cell_size)
        self.food = Food(random.randint(0, width // cell_size - 1) * cell_size,
                         random.randint(0, height // cell_size - 1) * cell_size, (255, 0, 0), cell_size)
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.model = self.load_model() or self.create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.exploration_rate = 1.0
        self.exploration_decay_rate = 0.999
        self.discount_factor = 0.95
        self.game_over = False
        self.memory = []
        self.memory_capacity = 10000
        self.batch_size = 32
        self.fps = 10

    def load_model(self):
        model_path = "snake/snake_model.h5"
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print("Loaded pre-trained model from snake_model.h5")
                return model
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        return None

    def create_model(self):
        grid_width = self.width // self.cell_size
        grid_height = self.height // self.cell_size
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(grid_width * grid_height + 2,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4)
        ])
        return model

    def get_state(self):
        grid_width = self.width // self.cell_size
        grid_height = self.height // self.cell_size
        grid = np.zeros((grid_height, grid_width))

        for segment in self.snake.body:
            x = segment[0] // self.cell_size
            y = segment[1] // self.cell_size
            if 0 <= x < grid_width and 0 <= y < grid_height:
                grid[y, x] = 1

        head_x, head_y = self.snake.body[0]
        head_x_cell = head_x // self.cell_size
        head_y_cell = head_y // self.cell_size

        apple_x = self.food.x // self.cell_size
        apple_y = self.food.y // self.cell_size

        # Calculate relative direction to apple
        dx = apple_x - head_x_cell
        dy = apple_y - head_y_cell
        direction_to_apple = np.array([dx, dy])

        # Normalize direction to apple
        if np.linalg.norm(direction_to_apple) > 0:
            direction_to_apple = direction_to_apple / np.linalg.norm(direction_to_apple)

        state = np.concatenate((grid.flatten(), direction_to_apple))

        return state

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            valid_actions = self.get_valid_actions()
            if not valid_actions:
                return None  # Handle no valid actions
            return random.choice(valid_actions)
        else:
            q_values = self.model(np.expand_dims(state, axis=0))
            q_values = q_values.numpy().flatten()

            valid_actions = self.get_valid_actions()
            if not valid_actions:
                return None  # Handle no valid actions
            q_values_masked = np.where(np.isin(np.arange(4), valid_actions), q_values, -np.inf)

            return np.argmax(q_values_masked)


    def get_valid_actions(self):
        head_x, head_y = self.snake.body[0]
        valid_actions = []
        for action in range(4):
            dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][action]
            new_head = (head_x + dx * self.cell_size, head_y + dy * self.cell_size)
            if 0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height and new_head not in self.snake.body:
                valid_actions.append(action)
        return valid_actions

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_value = tf.gather_nd(q_values, tf.stack([tf.range(self.batch_size), actions], axis=1))

            next_q_values = self.model(next_states)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)

            target = rewards + self.discount_factor * max_next_q_values * (1 - dones)
            loss = tf.reduce_mean(tf.square(target - q_value))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def reset_game(self):
        self.snake = Snake(self.width // 2, self.height // 2, (0, 255, 0), self.cell_size)
        self.food = Food(random.randint(0, self.width // self.cell_size - 1) * self.cell_size,
                         random.randint(0, self.height // self.cell_size - 1) * self.cell_size, (255, 0, 0), self.cell_size)
        self.score = 0
        self.game_over = False
        self.snake_length = 1
        if not self.get_valid_actions():
            self.reset_game()

    def run(self):
        running = True
        if not self.get_valid_actions():
            print("No valid starting position found. Try different game parameters.")
            return

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.fps = min(self.fps + 2, 30)
                    elif event.key == pygame.K_DOWN:
                        self.fps = max(self.fps - 2, 2)

            if self.game_over:
                self.reset_game()
                continue

            state = self.get_state()
            action = self.choose_action(state)

            if action is None:
                self.game_over = True
                self.reset_game()
                continue

            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.snake.move(directions[action])

            head_x, head_y = self.snake.body[0]
            head_rect = pygame.Rect(self.snake.body[0][0], self.snake.body[0][1], self.cell_size, self.cell_size)

            reward = 0
            done = False
            if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height or self.snake.check_collision():
                reward = -100
                self.game_over = True
                done = True
            elif head_rect.colliderect(self.food.get_rect()):
                reward = 50
                self.snake.grow()
                self.score += 1
                self.food.x = random.randint(0, self.width // self.cell_size - 1) * self.cell_size
                self.food.y = random.randint(0, self.height // self.cell_size - 1) * self.cell_size

            reward -= 1

            next_state = self.get_state()
            self.remember(state, action, reward, next_state, done)
            self.train_step()
            self.exploration_rate *= self.exploration_decay_rate

            self.screen.fill((0, 0, 0))
            self.snake.draw(self.screen)
            self.food.draw(self.screen)
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        self.model.save("snake/snake_model.h5")
        print("Model saved as snake_model.h5")

if __name__ == "__main__":
    game = Game(600, 400, 20)
    game.run()
