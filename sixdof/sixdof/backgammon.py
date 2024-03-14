import pygame
import random
import numpy as np

# Dimensions
BORDER_WIDTH = 40
GAP_WIDTH = 80
POINT_WIDTH = 80
POINT_HEIGHT = 240
SCREEN_WIDTH = 1120
SCREEN_HEIGHT = 640
CHECKER_RAD = 20
DIE_WIDTH = 40
PIP_RAD = 4
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 40

# Colors
WHITE = (128, 0, 128)
BLACK = (0, 255, 0)
TAN = (200, 160, 120)
BROWN = (110, 80, 50)
LIGHT_BROWN = (150, 130, 90)
DARK_BROWN = (120, 60, 0)
YELLOW = (240, 180, 0)

# Board
INIT_STATE = [2, 0, 0, 0, 0, -5,
              0, -3, 0, 0, 0, 5,
              -5, 0, 0, 0, 3, 0,
              5, 0, 0, 0, 0, -2]
POINT_LIM = 6

class Render:
    def __init__(self, game):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Verdana', 30)
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.game = game

    def update(self):
        flipped = pygame.transform.flip(self.screen, False, True)
        self.screen.blit(flipped, (0, 0))
        self.add_text()
        pygame.display.update()

    def add_text(self):
        text = self.font.render('Roll', True, (0, 0, 0))
        text_rect = text.get_rect(center = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

        if self.game.turn > 0:
            text = self.font.render('Robot\'s Turn', True, (0, 255, 0))
        else:
            text = self.font.render('Human\'s Turn', True, (128, 0, 128))
        text_rect = text.get_rect(center = (SCREEN_WIDTH / 2, BORDER_WIDTH / 2))
        self.screen.blit(text, text_rect)
    
    def draw(self):
        self.draw_background()
        self.draw_checkers()
        self.draw_dice()

    def draw_pips(self, die, pos):
        match (die):
            case 1:
                pygame.draw.circle(self.screen, BLACK, pos, PIP_RAD)
            case 2:
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
            case 3:
                pygame.draw.circle(self.screen, BLACK, pos, PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
            case 4:
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
            case 5:
                pygame.draw.circle(self.screen, BLACK, pos, PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
            case 6:
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] - DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1] + DIE_WIDTH / 4), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] - DIE_WIDTH / 4, pos[1]), PIP_RAD)
                pygame.draw.circle(self.screen, BLACK, (pos[0] + DIE_WIDTH / 4, pos[1]), PIP_RAD)
            case _:
                pass

    def draw_dice(self):
        pygame.draw.rect(self.screen, (200, 255, 200), ((SCREEN_WIDTH - BUTTON_WIDTH) / 2, (SCREEN_HEIGHT - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT))
        pygame.draw.rect(self.screen, DARK_BROWN, ((SCREEN_WIDTH - BUTTON_WIDTH) / 2, (SCREEN_HEIGHT - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT), 2)

        pygame.draw.rect(self.screen, WHITE, (SCREEN_WIDTH / 2 + DIE_WIDTH, (SCREEN_HEIGHT - DIE_WIDTH) / 2, DIE_WIDTH, DIE_WIDTH))
        pygame.draw.rect(self.screen, BLACK, (SCREEN_WIDTH / 2 + DIE_WIDTH, (SCREEN_HEIGHT - DIE_WIDTH) / 2, DIE_WIDTH, DIE_WIDTH), 2)
        pygame.draw.rect(self.screen, WHITE, (SCREEN_WIDTH / 2 - 2 * DIE_WIDTH, (SCREEN_HEIGHT - DIE_WIDTH) / 2, DIE_WIDTH, DIE_WIDTH))
        pygame.draw.rect(self.screen, BLACK, (SCREEN_WIDTH / 2 - 2 * DIE_WIDTH, (SCREEN_HEIGHT - DIE_WIDTH) / 2, DIE_WIDTH, DIE_WIDTH), 2)
        
        pos = (SCREEN_WIDTH / 2 - 1.5 * DIE_WIDTH, SCREEN_HEIGHT / 2)
        pos2 = (SCREEN_WIDTH / 2 + 1.5 * DIE_WIDTH, SCREEN_HEIGHT / 2)

        self.draw_pips(self.game.dice[0], pos)
        self.draw_pips(self.game.dice[1], pos2)

    def draw_checkers(self):
        point_pos = [SCREEN_WIDTH - BORDER_WIDTH + POINT_WIDTH / 2,
                     SCREEN_HEIGHT - BORDER_WIDTH - CHECKER_RAD]
        for i, point in enumerate(self.game.state[:24]):
            if i == 6:
                point_pos[0] -= GAP_WIDTH
            if i == 12:
                point_pos = [BORDER_WIDTH - POINT_WIDTH / 2, BORDER_WIDTH + CHECKER_RAD]
            if i == 18:
                point_pos[0] += GAP_WIDTH
            point_pos[0] -= POINT_WIDTH if i < 12 else -POINT_WIDTH
            for j in range(np.abs(point)):
                pos = point_pos.copy()
                pos[1] -= 2 * j * CHECKER_RAD if i < 12 else -2 * j * CHECKER_RAD
                color = WHITE if point > 0 else BLACK
                border = BLACK if point > 0 else WHITE
                pygame.draw.circle(self.screen, color, pos, CHECKER_RAD)
                pygame.draw.circle(self.screen, border, pos, CHECKER_RAD, 2)
        
        for i in range(self.game.state[24][0]):
            pos = (SCREEN_WIDTH / 2, SCREEN_HEIGHT - BORDER_WIDTH - (2 * i + 1) * CHECKER_RAD)
            pygame.draw.circle(self.screen, WHITE, pos, CHECKER_RAD)
            pygame.draw.circle(self.screen, BLACK, pos, CHECKER_RAD, 2)
        for i in range(self.game.state[24][1]):
            pos = (SCREEN_WIDTH / 2, BORDER_WIDTH + (2 * i + 1) * CHECKER_RAD)
            pygame.draw.circle(self.screen, BLACK, pos, CHECKER_RAD)
            pygame.draw.circle(self.screen, WHITE, pos, CHECKER_RAD, 2)

    def draw_background(self):
        pygame.draw.rect(self.screen, TAN, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, DARK_BROWN, (0, 0, SCREEN_WIDTH, BORDER_WIDTH))
        pygame.draw.rect(self.screen, DARK_BROWN, (0, 0, BORDER_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, DARK_BROWN, (0, SCREEN_HEIGHT - BORDER_WIDTH, SCREEN_WIDTH, BORDER_WIDTH))
        pygame.draw.rect(self.screen, DARK_BROWN, (SCREEN_WIDTH - BORDER_WIDTH, 0, BORDER_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, DARK_BROWN, ((SCREEN_WIDTH - GAP_WIDTH) / 2, 0, GAP_WIDTH, SCREEN_HEIGHT))
        
        shift = 0
        for i in range(12):
            if i > 5:
                shift = GAP_WIDTH
            color = BROWN if i % 2 else LIGHT_BROWN
            v1 = [[SCREEN_WIDTH - BORDER_WIDTH - POINT_WIDTH * i - shift, SCREEN_HEIGHT - BORDER_WIDTH],
                  [SCREEN_WIDTH - BORDER_WIDTH - POINT_WIDTH * (i + 1) - shift, SCREEN_HEIGHT - BORDER_WIDTH],
                  [SCREEN_WIDTH - BORDER_WIDTH - POINT_WIDTH * (i + 0.5) - shift, SCREEN_HEIGHT - BORDER_WIDTH - POINT_HEIGHT],
                  [SCREEN_WIDTH - BORDER_WIDTH - POINT_WIDTH * i - shift, SCREEN_HEIGHT - BORDER_WIDTH]]
            pygame.draw.polygon(self.screen, color, v1) 
            color = LIGHT_BROWN if i % 2 else BROWN
            v2 = [[BORDER_WIDTH + POINT_WIDTH * i + shift, BORDER_WIDTH],
                  [BORDER_WIDTH + POINT_WIDTH * (i + 1) + shift, BORDER_WIDTH],
                  [BORDER_WIDTH + POINT_WIDTH * (i + 0.5) + shift, BORDER_WIDTH + POINT_HEIGHT],
                  [BORDER_WIDTH + POINT_WIDTH * i + shift, BORDER_WIDTH]]
            pygame.draw.polygon(self.screen, color, v2)

            # if self.game.clicked is not None:
            #     if i == self.game.clicked:
            #         pygame.draw.polygon(self.screen, YELLOW, v1, 4) 
            #     if i == self.game.clicked - 12:
            #         pygame.draw.polygon(self.screen, YELLOW, v2, 4)

class Game:
    def __init__(self, state):
        self.state = state
        self.bar = [0, 0]
        self.state.append(self.bar)
        self.dice = [0, 0]
        self.turn = 1
        self.done = False
        self.clicked = None

    def roll(self):
        self.dice = np.random.randint(1, 7, size = 2).tolist()
    
    def move(self, point1, point2):
        if point1 is None:
            self.bar[0 if self.turn == 1 else 1] -= 1
        if self.turn == 1:
            if point2 is None:
                self.state[point1] -= 1
            elif self.state[point2] >= 0:
                if point1 is not None:
                    self.state[point1] -= 1
                self.state[point2] += 1
            elif self.state[point2] == -1:
                if point1 is not None:
                    self.state[point1] -= 1
                self.state[point2] = 1
                self.bar[1] += 1
        elif self.turn == -1:
            if point2 is None:
                self.state[point1] += 1
            elif self.state[point2] <= 0:
                if point1 is not None:
                    self.state[point1] += 1
                self.state[point2] -= 1
            elif self.state[point2] == 1:
                if point1 is not None:
                    self.state[point1] += 1
                self.state[point2] = -1
                self.bar[0] += 1
        self.state[24] = self.bar

    def is_valid(self, point1, point2, die, tried = False):
        if point1 is None:
            if (self.turn == 1 and -1 <= self.state[point2] < POINT_LIM and point2 + 1 == die or
                self.turn == -1 and -POINT_LIM < self.state[point2] <= 1 and point2 == 24 - die):
                return True
            return False
        if point2 is None:
            if (self.turn == 1 and self.state[point1] > 0 and (point1 + die >= 24 if tried else point1 + die == 24) or
                self.turn == -1 and self.state[point1] < 0 and (point1 - die <= -1 if tried else point1 - die == -1)):
                return True
            return False
        if (point1 == point2 or
            np.sign(self.state[point1]) != self.turn or
            self.state[point1] > 0 and (point1 > point2 or self.turn == -1) or
            self.state[point1] < 0 and (point1 < point2 or self.turn == 1) or
            abs(point1 - point2) != die):
            return False
        if (self.state[point1] > 0 and -1 <= self.state[point2] < POINT_LIM or
            self.state[point1] < 0 and -POINT_LIM < self.state[point2] <= 1):
            return True
        return False

    def possible_moves(self, die):
        moves = []

        # Move off bar
        if self.turn == 1 and self.bar[0]:
            for point in range(6):
                if self.is_valid(None, point, die):
                    moves.append((None, point))
            return moves
        elif self.turn == -1 and self.bar[1]:
            for point in range(18, 24):
                if self.is_valid(None, point, die):
                    moves.append((None, point))
            return moves
        
        # Move off board
        if self.all_checkers_in_end():
            if self.turn == 1:
                for point in range(18, 24):
                    if self.is_valid(point, None, die):
                        moves.append((point, None))
            elif self.turn == -1:
                for point in range(6):
                    if self.is_valid(point, None, die):
                        moves.append((point, None))

        # Normal moves
        if not moves:
            for point1 in range(24):
                for point2 in range(24):
                    if self.is_valid(point1, point2, die):
                        moves.append((point1, point2))

        # Move off board (again)
        if not moves and self.all_checkers_in_end():
            if self.turn == 1:
                for point in range(18, 24):
                    if self.is_valid(point, None, die, True):
                        moves.append((point, None))
            elif self.turn == -1:
                for point in range(6):
                    if self.is_valid(point, None, die, True):
                        moves.append((point, None))
    
        return moves
    
    def all_checkers_in_end(self):
        if self.turn == 1:
            for i in range(18):
                if self.state[i] > 0:
                    return False
            return True
        elif self.turn == -1:
            for i in range(6, 24):
                if self.state[i] < 0:
                    return False
            return True
        
    def num_checkers(self):
        if self.turn == 1:
            sum = self.bar[0]
            for point in self.state[:24]:
                if point > 0:
                    sum += point
            return sum
        if self.turn == -1:
            sum = self.bar[1]
            for point in self.state[:24]:
                if point < 0:
                    sum -= point
            return sum

# def area_triangle(x1, y1, x2, y2, x3, y3):
#     return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

# def in_triangle(vertices, pos):
#     x1, y1 = vertices[0]
#     x2, y2 = vertices[1]
#     x3, y3 = vertices[2]
#     x, y = pos

#     A = area_triangle(x1, y1, x2, y2, x3, y3)
#     A1 = area_triangle(x, y, x2, y2, x3, y3)
#     A2 = area_triangle(x1, y1, x, y, x3, y3)
#     A3 = area_triangle(x1, y1, x2, y2, x, y)
     
#     if A == A1 + A2 + A3:
#         return True
#     return False

# def get_point(pos):
#     shift = 0
#     for i in range(12):
#         if i > 5:
#             shift = GAP_WIDTH
#         vertices = [[SCREEN_WIDTH - BORDER_WIDTH - POINT_WIDTH * i - shift, BORDER_WIDTH],
#                     [SCREEN_WIDTH - BORDER_WIDTH - POINT_WIDTH * (i + 1) - shift, BORDER_WIDTH],
#                     [SCREEN_WIDTH - BORDER_WIDTH - POINT_WIDTH * (i + 0.5) - shift, BORDER_WIDTH + POINT_HEIGHT]]
#         if in_triangle(vertices, pos):
#             return i
        
#     shift = 0
#     for i in range(12):
#         if i > 5:
#             shift = GAP_WIDTH  
#         vertices = [[BORDER_WIDTH + POINT_WIDTH * i + shift, SCREEN_HEIGHT - BORDER_WIDTH],
#                     [BORDER_WIDTH + POINT_WIDTH * (i + 1) + shift, SCREEN_HEIGHT - BORDER_WIDTH],
#                     [BORDER_WIDTH + POINT_WIDTH * (i + 0.5) + shift, SCREEN_HEIGHT - BORDER_WIDTH - POINT_HEIGHT]]
#         if in_triangle(vertices, pos):
#             return i + 12
#     return None

# def on_button(pos):
#     return ((SCREEN_WIDTH - BUTTON_WIDTH) / 2 < pos[0] < (SCREEN_WIDTH + BUTTON_WIDTH) / 2 and
#             (SCREEN_HEIGHT - BUTTON_HEIGHT) / 2 < pos[1] < (SCREEN_HEIGHT + BUTTON_HEIGHT) / 2)

# def handle_click(game, pos):
#     point = get_point(pos)
#     if point is None:
#         if on_button(pos):
#             game.roll()
#         return
#     if game.clicked is None:
#         game.clicked = point
#     else:
#         game.move(game.clicked, point)
#         game.clicked = None

def choose_move(game, moves):
    # print("Legal moves: {}".format(moves))
    if moves:
        move = random.choice(moves)
        game.move(move[0], move[1])
        # print("Chosen move: {}".format(move))
    else:
        if not game.num_checkers():
            print("GAME OVER! {} WINS!".format("White" if game.turn == 1 else "Black"))
            game.done = True
        else:
            print("No legal moves.")

def handle_turn(game):
    game.roll()

    if game.dice[0] == game.dice[1]:
        for _ in range(4):
            moves = game.possible_moves(game.dice[0])
            choose_move(game, moves)
    else:
        larger = 1
        if game.dice[0] > game.dice[1]:
            larger = 0
        moves = game.possible_moves(larger)
        choose_move(game, moves)
        moves = game.possible_moves(not larger)
        choose_move(game, moves)

    game.turn *= -1

if __name__ == "__main__":
    game = Game(INIT_STATE)
    render = Render(game)

    while True:
        handle_turn(game)

        render.draw()
        render.update()

        if game.done:
            pygame.time.wait(10000)
            break

        pygame.time.wait(200)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     handle_click(game, pygame.mouse.get_pos())