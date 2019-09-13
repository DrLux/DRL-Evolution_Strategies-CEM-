import tensorflow as tf
import policy_network as network
import gym
import numpy as np

class planner(object):
    
    def __init__(self,env,num_plans):
        self.env = env
        self.plans = []
        self.num_plans = num_plans
        self.horizon = 100
        self.initial_plan()
        self.mean = []
        self.cov_matrix = np.eye(self.horizon)  
        
    def initial_plan(self):
        ## Extract info from env
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete) 
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]
    
        for i in range(self.num_plans):
            self.plans.append(network.policy_network(obs_dim, ac_dim,discrete))


    def evaluate_agents(self):
        planning_fitness = []
        for plan in self.plans:
            total_reward = 0 
            state = self.env.reset()
            done = False
            t = 0
            while not done:
                action = plan.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                done = done or (t >= self.horizon)
                state = next_state
                total_reward += reward

                t += 1 
            self.env.close()
            planning_fitness.append(total_reward)
        return sorted(planning_fitness, reverse=True)



    # TORNARE A OTTIMIZZARE
    def calculate_params(self):

        global_weights = []
        for pl in self.plans:
            global_weights.append(pl.get_weights())
        
        print(len(global_weights[0][0][0]))


'''
    def populate_new_generation(self, best_parents, top_limit):
        selected_idx_from_best = 0
        new_population = []

        # Take the best of his generation without noise
        new_population.append(self.plans[best_parents[-1]])

        
        for p in range(1,len(self.plans)):
            selected_idx_from_best =  np.random.choice(best_parents.shape[0], 1, replace=True)[0]
            new_population.append(self.plans[selected_idx_from_best])
            new_population[p].mutate()

        self.plans = new_population



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
        return self.plans[0]
'''