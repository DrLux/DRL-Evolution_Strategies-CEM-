import gym
import numpy as np
# da eliminare
import time


def main():
    env = gym.make('CartPole-v0')
    ## Extract info from env
    ac_dim = env.action_space.n 
    size_population = 2
    horizon = 3
    
    population = init_population(ac_dim,horizon,size_population)
    print(np.round(population))
    
    #population = new_population(np.zeros(3), np.eye(3), 5)
    #population = np.around(population) # to discrete values

    '''
    max_steps_per_episode = 100000
    #env = wrappers.Monitor(env, "./video")

    state = env.reset()
    done = False
    t = 0
    while not done:
        #env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        done = done or (t >= max_steps_per_episode)
        state = next_state
        
        t += 1 
    env.close()
    '''

# mean_plan = shape(horizon) with element in range (0,ac_dim)
# cov = shape(horizon,horizon) 
# size_population = scalar, number or plans in generation
# horizion = scalta, the lenght of each plan 
def new_population(mean,cov,horizon,size_population):
    return np.random.multivariate_normal(mean,cov,(horizon,size_population)) 

def init_population(ac_dim,horizon,size_population):
    random_plan = np.random.randint(low=0, high=ac_dim, size=horizon)
    return new_population(random_plan,np.eye(horizon),horizon,size_population)
    


if __name__ == "__main__":
    main()