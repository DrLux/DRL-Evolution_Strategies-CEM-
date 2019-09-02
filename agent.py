import tensorflow as tf
import policy_network as network
import gym
import numpy as np



class population(object):
    
    def __init__(self,env,num_agents):
        self.env = env
        self.population = []
        self.num_evaluation_run = 4 
        self.num_agents = num_agents
        self.max_steps_per_episode = 500
        self.init_population()
        
        # The best of his generation
        #elite_index = None

        # How many top agents to consider as parents
        #top_limit = 20

        # initialize N number of agents
        #num_agents = 500

        # run evolution until X generations
        #generations = 1000


    # Generate the initial population (one policy for each agent)
    def init_population(self):
        
        ## Extract info from env
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete) 
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]
    
        for i in range(self.num_agents):
            self.population.append(network.policy(obs_dim, ac_dim,discrete))


    def evaluate_agents(self):
        population_fitness = []
        for agent in self.population:
            total_reward = 0 
            for r in range(self.num_evaluation_run):
                state = self.env.reset()
                done = False
                t = 0
                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    done = done or (t >= self.max_steps_per_episode)
                    state = next_state
                    total_reward += reward

                    t += 1 
                self.env.close()
            population_fitness.append(total_reward / self.num_evaluation_run)
        return population_fitness



    def populate_new_generation(self, best_parents, top_limit):
        selected_idx_from_best = 0
        new_population = []

        # Take the best of his generation without noise
        new_population.append(self.population[best_parents[-1]])

        
        for p in range(1,len(self.population)):
            selected_idx_from_best =  np.random.choice(best_parents.shape[0], 1, replace=True)[0]
            new_population.append(self.population[selected_idx_from_best])
            new_population[p].mutate()

        self.population = new_population



    def evolve(self,top_limit,generations):
        for gen in range(generations):
            print("*** Generation: ", gen, "****")

            # return rewards of agents
            parent_fitness = self.evaluate_agents() #return average of 3 runs

            print("Best rewards: ")
            print(np.sort(parent_fitness)[-top_limit:])
            print(np.mean(parent_fitness[-top_limit:]))

            # sort by rewards
            # argsort ti da gli indici in ordine crescente, tu prendi gli indici degli ultimi top_limit --> np.argsort(rewards)[-top_limit:]
            best_parents = np.sort(np.argsort(parent_fitness)[-top_limit:]) 
            #best parent contiene gli indici migliori ma in ordine crescente
            
            self.populate_new_generation(best_parents,top_limit)
        return self.population[0]