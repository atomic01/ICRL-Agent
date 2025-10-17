import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class car_environment:
    
    image_height = 600
    image_width = 800
    camera_frame_rate = 0.5
    no_render_mode_status = False
    actor_list = []
    time_step = 0.05 # = 20fps
    synchronous_mode = True

    #higher level functions
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.load_world('Town06')
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = self.no_render_mode_status
        self.settings.fixed_delta_seconds = self.time_step
        self.settings.synchronous_mode = self.synchronous_mode
        self.world.apply_settings(self.settings)       
        self.save_video_to_disk = False
        self.blueprint_library = self.world.get_blueprint_library()
         
    def reset_environment(self):
        self.destroy_actors()#get rid of previous actors      
        self.collision_hist = []
        self.camera_hist = []
        self.actor_list = []
        
        self.place_vehicle_to_spawn_point()
        if self.save_video_to_disk:
            self.set_vehicle_camera()        
        self.set_collision_sensor()
        
    def step(self, action):
        if action[0] == 0 and action[1] == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=-0.015))
        elif action[0] == 0 and action[1] == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=-0.035))
       
        elif action[0] == 1 and action[1] == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.015))
        elif action[0] == 1 and action[1] == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.035))
        self.world.tick()#telling the server to advance for one frame, should consider waiting for a response
        vehicle_location = self.vehicle.get_location()
        x = int(vehicle_location.x)
        y = int(vehicle_location.y)
        state = np.array([[x, y]])
        
        if len(self.collision_hist) != 0:
            done = True
            distance_to_goal = int(np.sqrt((-364 - x)**2 + (80-y)**2))
            reward = -1000 - distance_to_goal
            state = np.array([[]])
        else:
            if x <= -354 and y >= 80:
                done = True
                reward = 1000
            else:
                done = False
                reward = 0

        return state, reward, done
           
    def test_agent(self, agent):
        self.reset_environment()
        done = False
        state = self.get_state()
        total_reward = 0
        while not done:
            action, y_predictions = agent.get_next_action(state, 0)
            state, reward, done = self.step(action)
            total_reward += reward
            
        if total_reward > 0:
            return True
        else:
            return False
        
    def save_video(self):
        self.save_video_to_disk = True
    
    #lower level functions
    def place_vehicle_to_spawn_point(self):
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.spawn_point = self.world.get_map().get_spawn_points()[43]
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
        self.location = self.vehicle.get_location()
        self.location.y -= 18
        self.location.x -= 255 #location.y += 65 location.x -= 365 <----- goal
        self.vehicle.set_location(self.location)
        self.actor_list.append(self.vehicle)
        
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=-300, y=65, z=120), carla.Rotation(pitch=-90)))
        
    def set_vehicle_camera(self):
        self.camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", f"{self.image_width}")
        self.camera_bp.set_attribute("image_size_y", f"{self.image_height}")
        self.camera_bp.set_attribute('sensor_tick', f"{self.camera_frame_rate}")
        self.camera_bp.set_attribute("fov", "110")
        self.camera_transform = carla.Transform(carla.Location(x=-5, z=3))
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame) )
        
    def set_collision_sensor(self):
        self.collision_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        self.CS_spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.collision_sensor = self.world.spawn_actor(self.collision_sensor_bp, self.CS_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
    
    def destroy_actors(self):
        if self.actor_list:
            for actor in self.actor_list:
                actor.destroy()
                
    def get_state(self):
        self.vehicle_location = self.vehicle.get_location()
        state = np.array([[int(self.vehicle_location.x), int(self.vehicle_location.y)]])
        
        return state
              
                  
class DQNAgent():
    learning_rate = 0.001
    discount_factor = 0.95
    epsilon = 0.99
    epsilon_decay = 0.995
    min_epsilon = 0.10
    number_of_episodes = 150
    refactor_interval = 100
    
            
    #higher level functions
    def __init__(self):
        self.model = self.create_model()
        self.opt = tf.optimizers.Adam(lr=self.learning_rate)
        self.episode = 0
        self.reward_array = []
        self.V = []
        self.throttle_V = []
        self.steering_V = []

                      
    def get_next_action(self, state, epsilon):
        state = self.normalize_state(state)
        y_pred = self.model(state)
        buffer = y_pred.numpy()
        buffer_throttle = []
        buffer_steer = []
        action = []
        #seperate decisions for throttle and steering
        for i in range(0,4):
            if i < 2:
                buffer_throttle.append(buffer[0][i])
            else:
                buffer_steer.append(buffer[0][i])                         
        
        #throttle decision
        if np.random.uniform() > epsilon:
            action.append(np.argmax(buffer_throttle))
        else:
            action.append(np.floor(np.random.uniform(0,2)).astype(np.int32))
        
        #steering decision
        if np.random.uniform() > epsilon:
            action.append(np.argmax(buffer_steer))
        else:
            action.append(np.floor(np.random.uniform(0,2)).astype(np.int32))
        
        return action, y_pred 
        
    def get_max_predictions(self, y_pred):
        buffer = y_pred.numpy()
        max_predictions = []
        buffer_throttle = []
        buffer_steer = []
        
        #seperate decisions for throttle and steering
        for i in range(0,4):
            if i < 2:
                buffer_throttle.append(buffer[0][i])
            else:
                buffer_steer.append(buffer[0][i])     
        max_predictions.append(np.argmax(buffer_throttle))
        max_predictions.append(np.argmax(buffer_steer))
            
        return max_predictions
        
   
    #lower level funcitions
    def create_model(self):
        inputs = tf.keras.layers.Input(shape=(2))
        skip = tf.keras.layers.Dense(20, activation='relu')(inputs)
        for i in range(4):
            layer = tf.keras.layers.Dense(20,activation='relu')(skip)
            skip = tf.keras.layers.Add()([skip, layer])
        outputs = tf.keras.layers.Dense(4,activation='linear')(skip)
        model = tf.keras.Model(inputs, outputs)
        return model

    def loss_func(self, inputs, value):
        with tf.GradientTape() as tape:
            y_pred = self.model(inputs)
            loss = tf.reduce_sum(tf.square(y_pred-value))
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads
    
    def normalize_state(self, state):
        x = (int(state[0][0]) + 374) / 124
        y = (int(state[0][1]) + 25) / 115
        state = np.array([[x, y]])
        
        return state
    
    
    def print_hyperparameters(self):
        print("Discount factor=", self.discount_factor, " Epsilon=", self.epsilon, " Epsilon decay=", self.epsilon_decay, " Learning rate=", self.learning_rate, " Refactorines=", self.refactor_interval )
    

def should_start_testing(reward_array):
    i = 0
    for reward in reward_array:
        if reward > 0:
            i += 1
    if i > 0:
        return True
    else:
        return False
    
def plot(reward_array, epoch, loss_array):
   
    plt.clf()
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.title("Training epoch: " + str(epoch) + "\nLosses")
    plt.plot(loss_array)
    
    plt.subplot(1,2,2)
    plt.title("Rewards")
    plt.plot(reward_array)
    plt.show()
    plt.pause(0.001)   
    
    
def main(): 
    finished_learning_episode = []
    epoch_loss_array = []
    exec_times_array = []
    number_of_fits_per_epoch = []    
    for epoch in range(1, 11): 
        agent = DQNAgent()
        env = car_environment()    
        reward_array = []
        loss_array = []
        episode = 0
        number_of_fits = 0        
        execution_start_time = time.time()        
        while episode in range(agent.number_of_episodes):    
            env.reset_environment()
            episode_reward = 0
            control_counter = 0
            done = False
            state = env.get_state()
            step = 0       
            while not done:
                action, y_predictions = agent.get_next_action(state, agent.epsilon)
                if control_counter % agent.refactor_interval == 0:
                     ic_old_y_predictions = y_predictions.numpy().copy()
                     ic_old_action = action.copy()
                     ic_old_state = agent.normalize_state(state)                
                old_state = state.copy()
                old_action = action.copy()   
                
                state, reward, done = env.step(old_action)                
                if not state.any():
                    state = old_state.copy()
                
                if control_counter % agent.refactor_interval == 0 or done == True:
                    y_predictions = agent.model(agent.normalize_state(state))
                    max_predictions = agent.get_max_predictions(y_predictions)
                    
                    throttle_value = reward + agent.discount_factor * max_predictions[0]
                    agent.throttle_V.append(throttle_value.squeeze())
                    throttle_value = throttle_value - np.mean(agent.throttle_V)
                    throttle_value = throttle_value / (np.std(agent.throttle_V) + 0.0000001)
                    
                    steering_value = reward + agent.discount_factor * max_predictions[1]
                    agent.steering_V.append(steering_value.squeeze())
                    steering_value = steering_value - np.mean(agent.steering_V)
                    steering_value = steering_value / (np.std(agent.steering_V) + 0.0000001)
                    
                    y_target = ic_old_y_predictions.copy()
                    y_target[0][ic_old_action[0]] = throttle_value
                    y_target[0][ic_old_action[1] + 2] = steering_value
                    
                    loss, grads = agent.loss_func(ic_old_state, y_target)
                    agent.opt.apply_gradients(zip(grads, agent.model.trainable_variables))
                    control_counter = 0
                    number_of_fits += 1
                       
                #metrics variables-------------------------------------------------------
                episode_reward += reward  
                step += 1
                control_counter += 1
                #------------------------------------------------------------------------
             
            if agent.epsilon > agent.min_epsilon:
                agent.epsilon *= agent.epsilon_decay              
            reward_array.append(episode_reward)
            loss_array.append(loss)
            print(episode,"R:", episode_reward, " steps: ", step,  "eps:", agent.epsilon , "fits: ", number_of_fits, " epoch:", epoch )      
            
            #TESTING THE MODEL
            testing_permision = should_start_testing(reward_array)
            if testing_permision:
                successful = env.test_agent(agent)
                if successful:
                    print("Testing outcome: --> Successful! ")
                    env.save_video()
                    done = env.test_agent(agent)  
                    last_episode = episode
                    episode = agent.number_of_episodes
                else:
                    print("Testing outcome: --> Failed! ")
                    last_episode = episode
                    
            episode += 1   
            if episode >= agent.number_of_episodes:
                plot(reward_array, epoch, loss_array)
                finished_learning_episode.append(last_episode-1)
                number_of_fits_per_epoch.append(number_of_fits)
                epoch_loss_array.append(loss)
                exec_times_array.append(time.time() - execution_start_time)
        
    number_of_fits_per_epoch = np.array(number_of_fits_per_epoch)
    finished_learning_episode = np.array(finished_learning_episode)
    epoch_loss_array = np.array(epoch_loss_array)
    exec_times_array = np.array(exec_times_array) 
    
    print("\nFits per epoch: ", number_of_fits_per_epoch, " mean: ", number_of_fits_per_epoch.mean(), " std: ", number_of_fits_per_epoch.std())
    print("Episode training ended mean: ", finished_learning_episode.mean(), " std: ", finished_learning_episode.std())
    print("Exec time mean: ", exec_times_array.mean(), " std: ", exec_times_array.std())
    print("Loss mean: ", epoch_loss_array.mean(), " std: ", epoch_loss_array.std() )
    agent.print_hyperparameters()
    
    print('Destroying actors...')
    env.destroy_actors()
    print('Done.')
    
    agent.model.summary()

    


if __name__ == '__main__':

    main()
