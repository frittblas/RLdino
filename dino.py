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

SEED = 4    # Seed for random number generation, always same seed for reproducibility

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
num_episodes = 100
max_steps_per_episode = 60
batch_size = 32
discount_factor = 0.95
replay_memory = []

# game variables

player_x = 20
player_y = 20
target_x = WIDTH / 5
target_y = HEIGHT / 5
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

speed = 4.0
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
track = pygame.image.load("assets/track.png")
cloud = pygame.image.load("assets/cloud.png")

# rects
dino_rect = dino_img.get_rect()
dino_rect.x = 20
dino_rect.y = HEIGHT / 2

# obstacle list
obstacles = []

clock = pygame.time.Clock()

# obstacle class

class Obstacle:
    def __init__(self, image, x, y):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

# game functions

def reset_game():
    global game_over, x_pos_bg, points, dino_rect
    global is_jumping, is_ducking, jump_vel, speed
    
    x = dino_rect.x
    y = dino_rect.y
    dino_rect = dino_dead_img.get_rect()
    dino_rect.x = x
    dino_rect.y = y
    
    screen.fill(BLACK)
    background()
    update_display(False)
    time.sleep(2)
    dino_rect = dino_img.get_rect()
    dino_rect.x = 20
    dino_rect.y = HEIGHT / 2
    
    x_pos_bg = 0
    game_over = False
    points = 0
    is_jumping = False
    is_ducking = False
    jump_vel = JUMP_VEL
    speed = 4.0
    
    
    obstacles.clear()
    
    random.seed(SEED)
    
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
            place_obstacles()

def place_obstacles():
    
    # generate random num between 1 and 3
    num = random.randint(1, 5)
    
    if num == 1:
        obstacles.append(Obstacle(cactus1_img, WIDTH, HEIGHT / 2 - 1))
    elif num == 2:
        obstacles.append(Obstacle(cactus2_img, WIDTH, HEIGHT / 2 - 1))
    elif num == 3:
        obstacles.append(Obstacle(cactus3_img, WIDTH, HEIGHT / 2 + 11))
    else:
        num2 = random.randint(1, 2)
        if num2 == 1:
            obstacles.append(Obstacle(ufo_img, WIDTH, HEIGHT / 2 - 20))
        else:
            obstacles.append(Obstacle(ufo_img, WIDTH, HEIGHT / 2 - 58))
        
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
    image_width = track.get_width()
    screen.blit(track, (x_pos_bg, y_pos_bg))
    screen.blit(track, (image_width + x_pos_bg, y_pos_bg))
    if x_pos_bg <= -image_width:
        screen.blit(track, (image_width + x_pos_bg, y_pos_bg))
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

def get_current_state():
    return np.array([player_x, player_y, target_x, target_y])

def execute_action(action):
    global player_x, player_y
    if action == 0:
        player_x -= player_speed
    elif action == 1:
        player_x += player_speed
    elif action == 2:
        player_y -= player_speed
    elif action == 3:
        player_y += player_speed

def get_reward():
    global player_x, player_y, player_size, target_x, target_y
    if (
        player_x < target_x + player_size
        and player_x + player_size > target_x
        and player_y < target_y + player_size
        and player_y + player_size > target_y
    ):
        return 1  # Positive reward if the player reaches the target
    
    # Check for collision with obstacles (add your obstacle conditions here)
    elif (
        # Add conditions for collision with obstacles
        # For example, if the player collides with the screen boundaries
        player_x < 0
        or player_x + player_size > WIDTH
        or player_y < 0
        or player_y + player_size > HEIGHT
    ):
        return -1  # Negative reward for running into an obstacle

    else:
        return 0  # No reward otherwise
    
    
    
    
def play_game():

    global player_x, player_y, target_x, target_y, game_over, you_win
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
            if not is_jumping:
                is_jumping = True
                is_ducking = False
        if keys[pygame.K_DOWN]:
            if not is_jumping:
                is_ducking = True
        else :
            is_ducking = False
            
            
        jump()
        if not is_jumping:
            duck()
            
        handle_obstacles()
        collide()
        logic()
        
        
        
        screen.fill(BLACK)
        background()
        update_display(True)

        if game_over:
            print("You Loose! Points: ", points)
            reset_game()

        clock.tick(FPS)
    
    
    

def train_model():

    global player_x, player_y, target_x, target_y, game_over, you_win
    global player_size, player_speed, points, game_over, you_win
    global replay_memory, epsilon

    for episode in range(num_episodes):
        reset_game()
        state = get_current_state()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Exploration-exploitation trade-off
            if np.random.rand() < epsilon:
                action = np.random.randint(4)  # Explore
            else:
                q_values = model.predict(state.reshape(1, -1))
                action = np.argmax(q_values)  # Exploit

            # Execute action in the environment
            execute_action(action)

            # Observe new state and reward
            new_state = get_current_state()
            reward = get_reward()

            # Store experience in replay memory
            replay_memory.append((state, action, reward, new_state))

            # Sample a random batch from replay memory for training
            batch = random.sample(replay_memory, min(batch_size, len(replay_memory)))
            states, actions, rewards, next_states = zip(*batch)

            # Compute target Q-values for training
            target_q_values = model.predict(np.array(states))
            next_q_values = model.predict(np.array(next_states))
            for i in range(len(batch)):
                target_q_values[i][actions[i]] = rewards[i] + discount_factor * np.max(next_q_values[i])

            # Train the model
            model.train_on_batch(np.array(states), target_q_values)

            total_reward += reward
            state = new_state
            
            update_display()

            #clock.tick(FPS)

        # Decay exploration rate
        epsilon *= epsilon_decay
        

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # Save the trained model
    model.save("trained_model.h5")




def use_model():
    global player_x, player_y, target_x, target_y, game_over, you_win
    global player_size, player_speed, points, game_over, you_win

    # Load the trained model
    loaded_model = load_model("trained_model.h5")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Use the model to predict the action
        state = get_current_state()
        q_values = loaded_model.predict(state.reshape(1, -1))
        action = np.argmax(q_values)

        # Execute action in the environment
        execute_action(action)

        # Observe new state and reward
        new_state = get_current_state()
        reward = get_reward()

        update_display()

        # Check for game over or win conditions
        if (
            player_x < target_x + player_size
            and player_x + player_size > target_x
            and player_y < target_y + player_size
            and player_y + player_size > target_y
        ):
            you_win = True

        if (
            player_x < 0
            or player_x > WIDTH
            or player_y < 0
            or player_y > HEIGHT
        ):
            game_over = True

        if game_over:
            print("Game Over!")
            reset_game()

        if you_win:
            points += 1
            print("You Win! Points: ", points)
            #reset_game()

        clock.tick(FPS)


def main():
    
    random.seed(SEED)
    
    obstacles.append(Obstacle(cactus1_img, WIDTH, HEIGHT / 2))
    #obstacles.append(Obstacle(cactus1_img, 500, HEIGHT / 2)) 
    #obstacles.append(Obstacle(cactus2_img, 900, HEIGHT / 2))
    
    play_game()
    #train_model()
    #use_model()

if __name__ == "__main__":
    main()
