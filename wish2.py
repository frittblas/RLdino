from keras import layers, models
from keras.models import load_model
import numpy as np
import pygame
import random
import sys

WIDTH, HEIGHT = 400, 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
FPS = 60

pygame.init()

# Define the neural network model
model = models.Sequential([
    layers.Dense(64, input_shape=(4,), activation='relu'),
    layers.Dense(4, activation='linear')  # 4 actions in the output layer
])

model.compile(optimizer='adam', loss='mse')

# Define exploration parameters
epsilon = 1.0  # exploration rate
epsilon_decay = 0.995  # decay rate for exploration

player_x = 20
player_y = 20
target_x = WIDTH / 5
target_y = HEIGHT / 5
game_over = False
you_win = False
player_size = 16
player_speed = 5
points = 0
game_over = False
you_win = False

# Training loop
num_episodes = 100
max_steps_per_episode = 60
batch_size = 32
discount_factor = 0.95
replay_memory = []

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Game")

clock = pygame.time.Clock()



def reset_game():
    global player_x, player_y, target_x, target_y, game_over, you_win
    #player_x, player_y = random.randint(0, WIDTH - player_size), random.randint(0, HEIGHT - player_size)
    #target_x, target_y = random.randint(0, WIDTH - player_size), random.randint(0, HEIGHT - player_size)
    player_x, player_y = 20, 20
    target_x, target_y = WIDTH / 5, HEIGHT / 5
    game_over, you_win = False, False

def go_left():
    global player_x
    player_x -= player_speed
    
def go_right():
    global player_x
    player_x += player_speed
    
def go_up():
    global player_y
    player_y -= player_speed
    
def go_down():
    global player_y
    player_y += player_speed


def update_display():
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (0, 0, WIDTH, HEIGHT), 2)
    pygame.draw.rect(screen, RED, (target_x, target_y, player_size, player_size))
    pygame.draw.rect(screen, GREEN, (player_x, player_y, player_size, player_size))
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
    

def train_model():

    global player_x, player_y, target_x, target_y, game_over, you_win
    global player_size, player_speed, points, game_over, you_win
    global replay_memory, epsilon

    for episode in range(num_episodes):
        reset_game()
        state = get_current_state()
        total_reward = 0

        #same_action_count = 3
        #same_action_counter = 0

        for step in range(max_steps_per_episode):
            
            #if same_action_counter == 0:
        # Exploration-exploitation trade-off
            if np.random.rand() < epsilon:
                action = np.random.randint(4)  # Explore
            else:
                q_values = model.predict(state.reshape(1, -1))
                action = np.argmax(q_values)  # Exploit

            # Execute action in the environment
            execute_action(action)
            #same_action_counter += 1
            #if same_action_counter == same_action_count:
                #same_action_counter = 0

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
            
            #update_display()

            #clock.tick(FPS)

        # Decay exploration rate
        epsilon *= epsilon_decay
        

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    # Save the trained model
    model.save("trained_model_v2.h5")





def play_game():

    global player_x, player_y, target_x, target_y, game_over, you_win
    global player_size, player_speed, points, game_over, you_win

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

        if keys[pygame.K_LEFT]:
            go_left()
        if keys[pygame.K_RIGHT]:
            go_right()
        if keys[pygame.K_UP]:
            go_up()
        if keys[pygame.K_DOWN]:
            go_down()
            

        if (
            player_x < target_x + player_size
            and player_x + player_size > target_x
            and player_y < target_y + player_size
            and player_y + player_size > target_y
        ):
            you_win = True

        # check collision with borders
        if (
            player_x < 0
            or player_x > WIDTH
            or player_y < 0
            or player_y > HEIGHT
        ):
            game_over = True

        update_display()

        if game_over:
            points = 0
            print("You Loose! Points: ", points)
            reset_game()

        if you_win:
            points += 1
            print("You Win! Points: ", points)
            reset_game()

        clock.tick(FPS)


def use_model():
    global player_x, player_y, target_x, target_y, game_over, you_win
    global player_size, player_speed, points, game_over, you_win

    # Load the trained model
    loaded_model = load_model("trained_model_v2.h5")

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
    #play_game()
    train_model()
    #use_model()

if __name__ == "__main__":
    main()