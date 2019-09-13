import planner as pl
import gym

def main():
    
    # Parameters of evolution

    # How many top agents to consider as parents
    top_limit = 20

    # initialize N number of agents
    num_plans = 10

    # run evolution until X generations
    generations = 5
    ############################################
    
    
    
    name_env = 'CartPole-v0'
    env = gym.make(name_env)
    planner = pl.planner(env,num_plans)
    #print(planner.evaluate_agents())
    print(planner.calculate_params())
    


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