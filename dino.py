import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras import layers, models
from keras.models import load_model
import numpy as np
import pygame
import random
import sys
import time

WIDTH, HEIGHT = 500, 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FPS = 60

JUMP_VEL = 6.0
GRAVITY = 0.4

MAX_SPEED = 25.0

SEED = 2    # Seed for random number generation, always same seed for reproducibility

FLAG_POS = WIDTH * 15

obstacle_rng = np.random.RandomState()

# Define the neural network model
model = models.Sequential([
    layers.Dense(64, input_shape=(4,), activation='relu'),
    layers.Dense(4, activation='linear')  # 4 actions in the output layer
])

model.compile(optimizer='adam', loss='mse')

# Define exploration parameters
epsilon = 1.0  # exploration rate
epsilon_decay = 0.995  # decay rate for exploration

# Training loop
num_episodes = 300
max_steps_per_episode = 600000
batch_size = 32
discount_factor = 0.95
replay_memory = []

# game variables

player_x = 20
player_y = 20
game_over = False
you_win = False
player_size = 16
player_speed = 5
is_jumping = False
is_ducking = False
points = 0
game_over = False
you_win = False

x_pos_bg = 0
y_pos_bg = HEIGHT / 2 + 20

speed = 5.0
jump_vel = JUMP_VEL

line_y = HEIGHT / 2 # ground line

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino skier")

# Set up the text
font = pygame.font.Font(None, 36)  # You can change the font and size as desired
text = font.render("Points: {}".format(points), True, WHITE)

# Load images
dino_img = pygame.image.load("assets/dino.png")
dino_duck_img = pygame.image.load("assets/dino_duck.png")
dino_dead_img = pygame.image.load("assets/dino_dead.png")
cactus1_img = pygame.image.load("assets/cactus1.png")
cactus2_img = pygame.image.load("assets/cactus2.png")
cactus3_img = pygame.image.load("assets/cactus3.png")
ufo_img = pygame.image.load("assets/ufo.png")
track_img = pygame.image.load("assets/track.png")
cloud_img = pygame.image.load("assets/cloud.png")
flag_img = pygame.image.load("assets/flag.png")

# rects
dino_rect = dino_img.get_rect()
dino_rect.x = 20
dino_rect.y = HEIGHT / 2

flag_rect = flag_img.get_rect()
flag_rect.y = HEIGHT / 2 - 65

# obstacle list
obstacles = []

# cloud list
clouds = []


clock = pygame.time.Clock()

# obstacle class
class Obstacle:
    def __init__(self, image, x, y):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        
# cloud class
class Cloud:
    def __init__(self, x, y):
        self.image = cloud_img
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        

# game functions

def reset_game():
    global game_over, you_win, x_pos_bg, points, dino_rect, flag_rect
    global is_jumping, is_ducking, jump_vel, speed
    
    x = dino_rect.x
    y = dino_rect.y
    dino_rect = dino_dead_img.get_rect()
    dino_rect.x = x
    dino_rect.y = y
    
    screen.fill(BLACK)
    background()
    if you_win:
        update_display(True)
        time.sleep(3)
    else:
        update_display(False)
        time.sleep(1)
    dino_rect = dino_img.get_rect()
    dino_rect.x = 20
    dino_rect.y = HEIGHT / 2
    
    x_pos_bg = 0
    game_over = False
    you_win = False
    points = 0
    is_jumping = False
    is_ducking = False
    jump_vel = JUMP_VEL
    speed = 4.0
    
    flag_rect.x = FLAG_POS
    
    obstacles.clear()
    
    obstacle_rng.seed(SEED)
    
    obstacles.append(Obstacle(cactus1_img, WIDTH, HEIGHT / 2))


def jump():
    global is_jumping, jump_vel, dino_rect

    if is_jumping:
        dino_rect.y -= jump_vel * 3
        jump_vel -= GRAVITY
    if jump_vel < -JUMP_VEL:
        is_jumping = False
        jump_vel = JUMP_VEL
        
    
def duck():
    global is_ducking, dino_rect
    
    if not is_jumping:
    
        if is_ducking:
            dino_rect = dino_duck_img.get_rect()
            dino_rect.x = 20
            dino_rect.y = HEIGHT / 2 + 20
        else:
            dino_rect = dino_img.get_rect()
            dino_rect.x = 20
            dino_rect.y = HEIGHT / 2
            
        #print("Ducking: ", is_ducking)

def handle_obstacles():
    # Iterate over obstacles
    for obstacle in obstacles:
        obstacle.rect.x -= speed
        if obstacle.rect.right < 0:
            obstacles.remove(obstacle)
            # if the obstacle list is empty, place a new obstacle
            if len(obstacles) == 0:
                place_obstacles()
            
def handle_flag():
    global you_win, game_over
    flag_rect.x -= speed
    if dino_rect.colliderect(flag_rect):
        you_win = True
        game_over = True

def place_obstacles():
    
    x_offset = 0
    
    iterations = 1
    
    if obstacle_rng.randint(1, 3) == 1:
        iterations = 2
    
    # for each iteration, place an obstacle
    for i in range(iterations):
    
        # generate random num between 1 and 3
        num = obstacle_rng.randint(1, 5)
        
        if num == 1:
            obstacles.append(Obstacle(cactus1_img, WIDTH + x_offset, HEIGHT / 2 - 1))
        elif num == 2:
            obstacles.append(Obstacle(cactus2_img, WIDTH + x_offset, HEIGHT / 2 - 1))
        elif num == 3:
            obstacles.append(Obstacle(cactus3_img, WIDTH + x_offset, HEIGHT / 2 + 11))
        else:
            num2 = obstacle_rng.randint(1, 3)
            if num2 == 1:
                obstacles.append(Obstacle(ufo_img, WIDTH + x_offset, HEIGHT / 2 - 20))
            else:
                obstacles.append(Obstacle(ufo_img, WIDTH + x_offset, HEIGHT / 2 - 96))
                
        x_offset = WIDTH / 2 + obstacle_rng.randint(-20, 20)

        
        
def logic():
    global speed, points
    #increase speed
    speed += 0.001
    if speed > MAX_SPEED:
        speed = MAX_SPEED
    
    #increase points slowly but no fractions are printed
    points += 0.1
    
        
def background():
    global x_pos_bg, y_pos_bg
    image_width = track_img.get_width()
    screen.blit(track_img, (x_pos_bg, y_pos_bg))
    screen.blit(track_img, (image_width + x_pos_bg, y_pos_bg))
    if x_pos_bg <= -image_width:
        screen.blit(track_img, (image_width + x_pos_bg, y_pos_bg))
        x_pos_bg = 0
    x_pos_bg -= speed

def collide():
    global game_over
    for obstacle in obstacles:
        if dino_rect.colliderect(obstacle.rect):
            game_over = True
            break

def update_display(alive):
    
    global text, dino_rect
    
    text = font.render("Points: {}".format(int (points)), True, WHITE)
    
    # Draw the obstacles
    for obstacle in obstacles:
        screen.blit(obstacle.image, obstacle.rect)
        
    # Draw the flag
    screen.blit(flag_img, flag_rect)
        
    if alive:
        # Draw the dino
        if is_ducking:
            screen.blit(dino_duck_img, dino_rect)
        else:
            screen.blit(dino_img, dino_rect)
    
    else:
        screen.blit(dino_dead_img, dino_rect)
    
    # Draw the text in the top right corner
    text_rect = text.get_rect()
    text_rect.x = WIDTH - 150
    screen.blit(text, text_rect)
    
    pygame.display.flip()

def execute_action(action):
    global is_jumping, is_ducking, speed
    # Action 0 is jump, action 1 is duck, action 2 is do nothing
    if action == 0:
        if not is_jumping and not is_ducking:
            is_jumping = True
    elif action == 1:
        if not is_jumping and not is_ducking:
            is_ducking = True
            speed = speed / 2
    elif action == 2:
        if is_ducking:
            is_ducking = False
            speed = speed * 2
        

    
def play_game():

    global player_x, player_y, game_over, you_win
    global player_size, player_speed, points, game_over, you_win
    global is_jumping, is_ducking
    global dino_rect, speed

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            execute_action(0)
        if keys[pygame.K_DOWN]:
            execute_action(1)
        else :
            execute_action(2)
            
            
        jump()
        if not is_jumping:
            duck()
            
        handle_obstacles()
        handle_flag()
        collide()
        logic()
        
        
        
        screen.fill(BLACK)
        background()
        update_display(True)

        if game_over:
            if you_win:
                print("You Win! Points: ", points)
            else:
                print("You Loose! Points: ", points)
            reset_game()

        clock.tick(FPS)
    
    
    

def get_current_state():
    # Return the current state of the game
    return np.array([dino_rect.x, dino_rect.y, obstacles[0].rect.x, obstacles[0].rect.y])

def get_reward():
    # Positive reward for staying alive, larger negative reward for dying
    if game_over:
        return -100
    else:
        return 1

def train_model_v2():
    global epsilon

    #model = load_model('dino_skier_model.h5')
    #epsilon = 0.22229219984074702

    for episode in range(num_episodes):
        reset_game()
        state = get_current_state()

        for step in range(max_steps_per_episode):
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            if not is_jumping:
            
                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, 4)  # Explore: choose a random action
                else:
                    q_values = model.predict(state.reshape(1, 4))
                    action = np.argmax(q_values[0])  # Exploit: choose the action with the highest q-value

                execute_action(action)
                next_state = get_current_state()
                reward = get_reward()
                model.fit(state.reshape(1, 4), np.array([reward]), epochs=1, verbose=0)
                state = next_state

            jump()            
            duck()
            handle_obstacles()
            handle_flag()
            collide()
            logic()
            
            screen.fill(BLACK)
            background()
            update_display(True)

            if game_over:
                if you_win:
                    print("You Win! Points: ", points)
                else:
                    print("You Loose! Points: ", points)
                reset_game()
                break

            #clock.tick(FPS)
            time.sleep(0.01)

        epsilon *= epsilon_decay  # Decrease the exploration rate
        
        print("Episode: ", episode, " Epsilon: ", epsilon)
        
    model.save('dino_skier_model.h5')
        

def use_model():
    # Load the trained model
    trained_model = load_model('dino_skier_model.h5')

    # Reset the game and get the initial state
    reset_game()
    state = get_current_state()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Use the trained model to predict the action
        q_values = trained_model.predict(state.reshape(1, 4))
        action = np.argmax(q_values[0])

        # Perform the action
        execute_action(action)
        state = get_current_state()

        jump()
        duck()
        handle_obstacles()
        handle_flag()
        collide()
        logic()

        screen.fill(BLACK)
        background()
        update_display(True)

        #clock.tick(FPS)

    if you_win:
        print("You Win! Points: ", points)
    else:
        print("You Lose! Points: ", points)


def main():
    
    #random.seed(SEED)
    obstacle_rng.seed(SEED)
    
    obstacles.append(Obstacle(cactus1_img, WIDTH, HEIGHT / 2))


    # place flag far to the right
    flag_rect.x = FLAG_POS
        
    #clouds.append(Cloud(WIDTH / 2, HEIGHT / 4))
    #clouds.append(Cloud(WIDTH / 4, HEIGHT / 6))
    #clouds.append(Cloud(WIDTH / 1.5, HEIGHT / 8))
    
   
    #play_game()
    #train_model_v2()
    use_model()

if __name__ == "__main__":
    main()
