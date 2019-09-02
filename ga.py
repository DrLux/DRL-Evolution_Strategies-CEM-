import agent 
import gym

def main():
    
    # Parameters of evolution

    # How many top agents to consider as parents
    top_limit = 10

    # initialize N number of agents
    num_agents = 100

    # run evolution until X generations
    generations = 20
    
    name_env = 'CartPole-v0'
    env = gym.make(name_env)
    agents = agent.population(env,num_agents)
    highlander = agents.evolve(top_limit,generations)

    play_env(highlander,env)



def play_env(policy,env):
    # Play trained policy
    max_steps_per_episode = 1000
    #env = wrappers.Monitor(env, "./video")

    state = env.reset()
    done = False
    t = 0
    while not done:
        env.render()
        #action = env.action_space.sample()
        action = policy.get_action(state)
        next_state, reward, done, _ = env.step(action)
        done = done or (t >= max_steps_per_episode)
        state = next_state

        t += 1 
    env.close()

if __name__ == "__main__":
    main()