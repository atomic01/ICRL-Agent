import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import time


environment_rows = 10
environment_columns = 10
N = 10

rewards = np.full((environment_rows, environment_columns), -100.)

maze = {} 
maze[0] = [3]
maze[1] = [i for i in range(1,9)]
maze[2] = [1, 6, 8]
maze[3] = [1, 2, 3, 5, 6, 8]
maze[4] = [1, 3, 5, 8]
maze[5] = [1, 2, 3, 4, 5, 7, 8]
maze[6] = [5]
maze[7] = [1, 3, 5, 8]
maze[8] = [i for i in range(1,9)]

for row_index in range(0,N-1):
    for column_index in maze[row_index]:
        rewards[row_index, column_index] = -1
rewards[9, 7] = 1000
        
    

    
data = np.zeros((environment_columns, environment_rows))
for j in range(environment_rows):
    for i in range(environment_columns):
        if rewards[j][i] == -100:
            data[j][i] = 0
        else:
            data[j][i] = 1
    


def loss_func(model,inputs,value):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.reduce_sum(tf.square(y_pred-value))
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads



def is_terminal_state(state):
    if state[0].any() < 0  or state[0].any() > (N-1):
        return True
    elif rewards[state[0][0], state[0][1]] == -100:
        return True
    else:
        return False
    

def get_next_action(state, epsilon):
    y_pred = model(state / (N-1))
    buffer = y_pred.numpy()
    
    done = False
    while done != True:
        new_state = state.copy()
        if np.random.uniform() > epsilon:
            action = np.argmax(buffer)
        else:
            action = np.floor(np.random.uniform(0,4)).astype(np.int32)
        
        new_state = get_next_state(state, action)
        if not is_terminal_state(new_state):
            done = True  
        else:
            buffer[0][action] = -100
            
    return action, y_pred


def get_next_state(old_state, action):
    state = old_state.copy()
    if action == 0: #up
        state[0][0] -= 1
    elif action == 1: #right
        state[0][1] += 1
    elif action == 2: #down
        state[0][0] += 1
    elif action == 3: #left
        state[0][1] -= 1
        
    if state[0][0] < 0 or state[0][1] < 0:
        reward = -100
    else:
        reward = rewards[state[0][0], state[0][1]]
    
    if reward == 1000:
        end = True
    else:
        end = False
        
    return state, reward, end


def draw_shortest_path(start, epoch, reward_array):
    shortest_path = get_shortest_path(start)
    if shortest_path:
        for i in shortest_path:
            data[i[0][0], i[0][1]] = 0.4
            
    for x in range(N):
        for y in range(N):
            s = np.array([[x,y]])
            maze_val[s[0][0],s[0][1]] = np.max(model(s/(N-1)),axis=1)
    
    plt.clf()
    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.title("training epoch: " + str(epoch))
    plt.imshow((maze_val - np.min(maze_val)) / (np.max(maze_val) - np.min(maze_val)))
    plt.subplot(1,3,2)
    plt.title("path lenght: " + str(len(shortest_path)))
    plt.imshow(data, interpolation='nearest', cmap='hot')
    plt.subplot(1,3,3)
    plt.plot(reward_array)
    plt.show()
    plt.pause(0.001)
        
    
def get_shortest_path(start):
    goal_reached = False
    if is_terminal_state(start):
        return []
    else: 
        shortest_path = []
        shortest_path.append(start)

        state = start.copy()
        while not is_terminal_state(state) and goal_reached == False:
            action, pred = get_next_action(state, 0)
            state = get_next_state(state, action)
            shortest_path.append(state)        
            if len(shortest_path) > lenght_of_the_shortest_path:
                return shortest_path
            
            reward = rewards[state[0][0], state[0][1]]
            if reward == 1000:
                goal_reached = True
                
        return shortest_path
    
    
def get_max_prediction(y_predictions, state):
    buffer = y_predictions.numpy()
    
    done = False
    while done != True:
        action = np.argmax(buffer)
        new_state = state.copy()
        new_state = get_next_state(state, action)
        if is_terminal_state(new_state):
            buffer[0][action] = -1000000
        else:
            done = True
            
    max_prediction = y_predictions[0][action]
    max_prediction = np.array([max_prediction])
    
    return max_prediction


def test(start):
    shortest_path = get_shortest_path(start)
    
    end = shortest_path[len(shortest_path) - 1]
    reward = rewards[end[0][0], end[0][1]]
    if reward == 1000:
        goal_reached = True
    else:
        goal_reached = False
        
        
    if len(shortest_path) < lenght_of_the_shortest_path and goal_reached == True:
        return True, len(shortest_path)
    else:
        return False, len(shortest_path)


finished_learning_episode = []
loss_array = []
exec_times_array = []
number_of_steps_array = []



for epoch in range(1, 11):    
    #initialize NN
    inputs = tf.keras.layers.Input(shape=(2))
    skip = tf.keras.layers.Dense(75, activation='relu')(inputs)
    for i in range(4):
        layer = tf.keras.layers.Dense(75,activation='relu')(skip)
        skip = tf.keras.layers.Add()([skip,layer])
    outputs = tf.keras.layers.Dense(4,activation='linear')(skip)
    model = tf.keras.Model(inputs,outputs)

    learning_rate = 0.001 
    opt = tf.optimizers.Adam(lr=learning_rate)
    discount_factor = 0.95
    epsilon = 0.99
    epsilon_decay = 0.99
    number_of_episodes = 500
    sampling_frequency = 6
    MAX_STEPS = 500
    maze_val = np.zeros((N,N))
    lenght_of_the_shortest_path = 20
    refactor_interval = 6
    
    #image pixel data
    original_data = data.copy()
    
    #metrics sotrage arrays
    reward_array = []
    V=[]
    test_result = False
    episode = 0
        
    start_time = time.time()
    
    if epoch == 1:
        info_string = "Discount factor=", discount_factor, " Epsilon=", epsilon, " Epsilon decay=", epsilon_decay, " Max steps=", MAX_STEPS, " Learning rate=", learning_rate, " Refactorines=", refactor_interval 

        
    while episode in range(number_of_episodes):
        start = np.array([[0,3]])
        state = start.copy()
        
        step = 0
        aggr_reward = 0
        end = False
        control_counter = 0
        
        while not is_terminal_state(state) and step < MAX_STEPS and not end:
            action, y_predictions = get_next_action(state, epsilon)
            
            if control_counter % refactor_interval == 0:
                ic_old_y_predictions = y_predictions.numpy().copy()
                ic_old_action = action.copy()
                ic_old_state = state.copy()
            
  
            old_action = action.copy()
            old_state = state.copy()
            
            state, reward, end = get_next_state(old_state, old_action)
                
            if control_counter % refactor_interval == 0 or end == True:
                y_predictions = model(state / (N-1))
                max_prediction = get_max_prediction(y_predictions, state)
                
                value = reward + discount_factor * max_prediction
                V.append(value.squeeze())
                value = value - np.mean(V)
                value = value / (np.std(V) + 0.0000001)
                
                y_target = ic_old_y_predictions.copy()
                y_target[0][ic_old_action] = value
                
                loss, grads = loss_func(model,ic_old_state/(N-1), y_target)
                opt.apply_gradients(zip(grads, model.trainable_variables))
                control_counter = 0
            
            #metrics variables-------------------------------------------------------
            aggr_reward += reward  
            step += 1
            control_counter += 1
            #------------------------------------------------------------------------
            
        if epsilon > 0.01:
            epsilon *= epsilon_decay
            
        reward_array.append(aggr_reward)
        print(episode,"R:", aggr_reward, " steps: ", step,  "eps:", epsilon, )
    
        test_result, training_steps = test(start)
        if test_result == True or episode==number_of_episodes-1:
            draw_shortest_path(start, epoch, reward_array)
            data = original_data.copy()            
            
            finished_learning_episode.append(episode)
            loss_array.append(loss)
            exec_times_array.append(time.time() - start_time)
            number_of_steps_array.append(training_steps)
            
            episode = number_of_episodes
            
        
        episode += 1            
        
        
        

finished_learning_episode = np.array(finished_learning_episode)
loss_array = np.array(loss_array)
exec_times_array = np.array(exec_times_array)
number_of_steps_array = np.array(number_of_steps_array)
reward_array = number_of_steps_array.copy()
for i in range(len(number_of_steps_array)):
    reward_array[i] = 1000 - number_of_steps_array[i]
    

#print(finished_learning_episode)
print("Episode training ended mean: ", finished_learning_episode.mean(), " std: ", finished_learning_episode.std())

#print(number_of_steps_array)
print("Steps mean: ", number_of_steps_array.mean(), " std: ", number_of_steps_array.std() )

#print(reward_array)
print("Rewards mean: ", reward_array.mean(), " std: ", reward_array.std() )

#print(exec_times_array)
print("Exec time mean: ", exec_times_array.mean(), " std: ", exec_times_array.std())

print("Loss mean: ", loss_array.mean(), " std: ", loss_array.std() )

print(info_string)
model.summary()
