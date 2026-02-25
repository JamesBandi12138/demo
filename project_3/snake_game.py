import pygame
import random
import sys


class Snake:
    def __init__(self, x, y):
        self.body = [(x, y)]
        self.direction = "RIGHT"
        self.grow = False

    def move(self):
        head_x, head_y = self.body[0]
        if self.direction == "UP":
            head_y -= 20
        elif self.direction == "DOWN":
            head_y += 20
        elif self.direction == "LEFT":
            head_x -= 20
        elif self.direction == "RIGHT":
            head_x += 20
        new_head = (head_x, head_y)
        self.body.insert(0, new_head)
        if not self.grow:
            self.body.pop()
        self.grow = False

    def change_direction(self, new_dir):
        if (new_dir == "UP" and self.direction != "DOWN" or
            new_dir == "DOWN" and self.direction != "UP" or
            new_dir == "LEFT" and self.direction != "RIGHT" or
            new_dir == "RIGHT" and self.direction != "LEFT"):
            self.direction = new_dir


class Food:
    def __init__(self):
        self.x = random.randint(0, 19) * 20
        self.y = random.randint(0, 19) * 20

    def respawn(self, snake_body):
        while True:
            self.x = random.randint(0, 19) * 20
            self.y = random.randint(0, 19) * 20
            if (self.x, self.y) not in snake_body:
                break


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("贪吃蛇")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("simhei", 24)
        self.reset()

    def reset(self):
        self.snake = Snake(200, 200)
        self.food = Food()
        self.score = 0
        self.game_over = False
        self.game_started = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if not self.game_started:
                    if event.key == pygame.K_SPACE:
                        self.game_started = True
                    continue
                if event.key == pygame.K_UP:
                    self.snake.change_direction("UP")
                elif event.key == pygame.K_DOWN:
                    self.snake.change_direction("DOWN")
                elif event.key == pygame.K_LEFT:
                    self.snake.change_direction("LEFT")
                elif event.key == pygame.K_RIGHT:
                    self.snake.change_direction("RIGHT")

    def check_collision(self):
        head = self.snake.body[0]
        if (head[0] < 0 or head[0] >= 400 or head[1] < 0 or head[1] >= 400):
            return True
        if head in self.snake.body[1:]:
            return True
        return False

    def update(self):
        if not self.game_started:
            return

        if self.game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.reset()
            return

        self.snake.move()
        head = self.snake.body[0]

        if head == (self.food.x, self.food.y):
            self.snake.grow = True
            self.score += 1
            self.food.respawn(self.snake.body)

        if self.check_collision():
            self.game_over = True

    def draw(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0),
                     (self.food.x, self.food.y, 20, 20))
        for segment in self.snake.body:
            pygame.draw.rect(self.screen, (0, 200, 0),
                         (segment[0], segment[1], 20, 20))

        if not self.game_started:
            start_text = self.font.render("按空格开始", True, (255, 255, 255))
            text_rect = start_text.get_rect(center=(200, 180))
            self.screen.blit(start_text, text_rect)
        else:
            score_text = self.font.render(f"得分: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))

            if self.game_over:
                over_text = self.font.render("游戏结束! 按空格重试", True, (255, 0, 0))
                text_rect = over_text.get_rect(center=(200, 200))
                self.screen.blit(over_text, text_rect)

        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(10)


if __name__ == "__main__":
    game = Game()
    game.run()
