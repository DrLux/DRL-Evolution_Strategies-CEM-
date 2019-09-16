from collections import deque
import numpy as np
import gym
import time
import matplotlib.pyplot as plt


# Using CEM to maximize the "evaluate_plan" result 
def update_param(theta_mean,theta_std,top_per,batch_size,env):
    #theta: list of "batch_size" vectors each one length "obs"
    #create a theta in mean and then add a std + gaussian noise
    #shape (batch_size, env.observation_space.shape[0])
    theta_sample = np.tile(theta_mean, (batch_size, 1)) + np.tile(theta_std, (batch_size, 1)) * np.random.randn(batch_size, theta_mean.size)
    reward_sample = np.array([evaluate_plan(env, th)[0] for th in theta_sample])

    # get the index of 2% best thetas
    top_idx = np.argsort(-reward_sample)[:int(np.round(batch_size * top_per))]
    top_theta = theta_sample[top_idx]

    #update theta
    theta_mean = top_theta.mean(axis = 0)
    theta_std = top_theta.std(axis = 0) 
    return theta_mean,theta_std

def evaluate_plan(env, theta):
    total_rewards = 0
    observation = env.reset()
    for t in range(env._max_episode_steps):
        action = get_action(observation, theta)
        observation, reward, done, _ = env.step(action)
        total_rewards += reward
        if done: break
    return total_rewards, t

# Define the policy network (just 1 layer) 
def get_action(ob,weights):
    W1 = weights[:-1]
    b1 = weights[-1]
    return int((ob.dot(W1) + b1) < 0)

# Core function to evolve agent according to Cross Entropy Method
def evolve_agent(env):
    GENERATIONS   = 100
    POPULATION    = 300
    elite         = 0.2 # percentage of the population we want preserve
    std           = 1 # scale of standard deviation

    plot_data = []

    # initialize parameters
    theta_mean = np.zeros(env.observation_space.shape[0] + 1)
    theta_std = np.ones_like(theta_mean) * std


    #Buffer list for statistics
    episode_history = deque(maxlen=100) 
    
    for itr in range(GENERATIONS):
        theta_mean,theta_std = update_param(theta_mean,theta_std,elite,POPULATION,env)
        total_rewards, t = evaluate_plan(env, theta_mean)

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)
        plot_data.append(mean_rewards)

        print("Episode {}".format(itr))
        print("Finished after {} timesteps".format(t+1))
        print("Reward for this episode: {}".format(total_rewards))
        print("Average reward for last 100 episodes: {}".format(mean_rewards))
        # Leaderboard treshold to win the env: https://github.com/openai/gym/wiki/CartPole-v0
        if mean_rewards >= 195.0:
            print("Environment solved after {} episodes".format(itr+1))
            break
    
    
    plt.plot(plot_data)
    plt.show()

    return theta_mean,theta_std

# Show an episode of the trained agent
def show_agent(env,theta_mean,theta_std):
    theta = theta_mean + (theta_std * np.random.randn(theta_mean.size))

    observation = env.reset()
    done = False
    total_rewards = 0
    
    while not done:
        action = get_action(observation, theta)
        observation, reward, done, _ = env.step(action)
        total_rewards += reward
        time.sleep(0.008)
        env.render()

    env.close()    
    print("Final Reward: ",total_rewards)


def main():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    theta_mean,theta_std = evolve_agent(env)
    show_agent(env,theta_mean,theta_std)

if __name__ == "__main__":
    main()