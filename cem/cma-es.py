from collections import deque
import numpy as np
import gym


'''
#CEM
# maximize function theta_rollout through cross-entropy method
def update_param(theta_mean,theta_std,top_per,batch_size,env,MAX_STEPS):
    #theta(weight of leyer 1): list of "batch_size" vectors each one length "obs" 
    #create a theta in mean and then add a std + gaussian noise
    #shape (batch_size, env.observation_space.shape[0])
    theta_sample = np.tile(theta_mean, (batch_size, 1)) + np.tile(theta_std, (batch_size, 1)) * np.random.randn(batch_size, theta_mean.size)
    reward_sample = np.array([evaluate_plan(env, th, MAX_STEPS)[0] for th in theta_sample])

    # get the index of 2% best thetas
    top_idx = np.argsort(-reward_sample)[:int(np.round(batch_size * top_per))]
    top_theta = theta_sample[top_idx]
    theta_mean = top_theta.mean(axis = 0)
    theta_std = top_theta.std(axis = 0) #cem
    return theta_mean,theta_std
'''


#CMA-ES
# maximize function theta_rollout through cross-entropy method
def update_param(theta_mean_vec,theta_cov_matrix,top_per,batch_size,env,MAX_STEPS,horizon):
    theta_sample = np.random.multivariate_normal(theta_mean_vec,theta_cov_matrix,(env.observation_space.shape[0] + 1,batch_size))
    print(theta_sample.shape)
    #reward_sample = np.array([evaluate_plan(env, th, MAX_STEPS)[0] for th in theta_sample])



def evaluate_plan(env, theta, num_steps, render = False):
    total_rewards = 0
    observation = env.reset()
    for t in range(num_steps):
        action = get_action(observation, theta)
        observation, reward, done, _ = env.step(action)
        total_rewards += reward
        if render: env.render()
        if done: break
    return total_rewards, t

# define policy neural network
def get_action(ob,weights):
    W1 = weights[:-1]
    b1 = weights[-1]
    return int((ob.dot(W1) + b1) < 0)

def main():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    ITERATIONS = 20
    MAX_STEPS    = 200
    batch_size   = 40
    top_per      = 0.2 # percentage of theta with highest score selected from all the theta
    std          = 1 # scale of standard deviation
    horizon      = 6

    # initialize
    theta_mean_vec = np.zeros(horizon)
    theta_cov_matrix = np.eye(horizon)


    episode_history = deque(maxlen=100) #init buffer list
    
    for itr in range(ITERATIONS):
        theta_mean_vec,theta_cov_matrix = update_param(theta_mean_vec,theta_cov_matrix,top_per,batch_size,env,MAX_STEPS,horizon)
        total_rewards, t = evaluate_plan(env, theta_mean_vec, MAX_STEPS, render = True)

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        print("Episode {}".format(itr))
        print("Finished after {} timesteps".format(t+1))
        print("Reward for this episode: {}".format(total_rewards))
        print("Average reward for last 100 episodes: {}".format(mean_rewards))
        if mean_rewards >= 195.0:
            print("Environment {} solved after {} episodes".format(env_name, itr+1))
            break
    env.close()

if __name__ == "__main__":
    main()