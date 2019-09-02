import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

class policy(object):

    def __init__(self,obs_dim,ac_dim,discrete):
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.discrete = discrete

        #aggiungerlo nella relazione
        n_layers = 2
        size = 64 #dimension of the hidden layer


        #define placeholder
        self.sy_ob_no = tf.placeholder(shape=[None, obs_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        else:
            self.sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) #mean and std
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv_n", dtype=tf.float32) 

        ###########
        # Actor
        ###########

        # Define policy forward pass
        if self.discrete:
            policy_parameters = self.build_mlp(self.sy_ob_no, ac_dim, "policy_forward_pass", n_layers, size) #build the policy nn
        else:
            sy_mean = self.build_mlp(self.sy_ob_no, ac_dim, "policy_forward_pass", n_layers, size) 
            sy_logstd = tf.get_variable(dtype=tf.float32, shape=[ac_dim], name="logstd")
            policy_parameters = (sy_mean, sy_logstd)

        #sample action from parameters
        self.sy_sampled_ac = self.sample_action(policy_parameters)

        #Calculate Function Loss
        #take log prob of the actions
        if self.discrete:
            sy_logits_na = policy_parameters
            
            # (batch_size, self.ac_dim) prob for each actions
            sy_prob = tf.nn.softmax(sy_logits_na) 
            
            #hot vector of tao actions
            tao_hot_actions = tf.one_hot(self.sy_ac_na, depth=ac_dim) 
            
            #total log probabilities
            sy_logprob_n = tf.log(tf.reduce_sum(sy_prob * tao_hot_actions, axis=1)) 
            
            #sice The log prob is equivalent to -(cross_entropy) you can do it with only 1 line
            #sy_logprob_n = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(sy_ac_na, depth=self.ac_dim), logits=sy_logits_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_logprob_n = tfp.distributions.MultivariateNormalDiag(sy_mean, tf.exp(sy_logstd)).log_prob(self.sy_ac_na)

        actor_loss = tf.reduce_sum(-sy_logprob_n * self.sy_adv_n)
        self.init_tf_sess()

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        
        # may need if using GPU
        tf_config.gpu_options.allow_growth = True 
        
        self.sess = tf.Session(config=tf_config)
        
        # Equivalent to `with self.sess:`
        self.sess.__enter__() 
        
        ## Once that sess.run finish it's execution nothing in memory keep lives except for the variables. So we need to reinitialize them 
        tf.global_variables_initializer().run() 


    def build_mlp(self,input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            for i in range(n_layers):
                
                # The first layer
                if i == 0: 
                    hidden_layers = tf.layers.dense(input_placeholder, size, activation=activation, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='input_layer')
                else:
                    hidden_layers = tf.layers.dense(hidden_layers,size, activation=activation, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='hidden-%d' % i)
            
            # The output layer    
            output_placeholder = tf.layers.dense(hidden_layers,output_size,activation=output_activation,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='output_layer')
        
        return output_placeholder

    def sample_action(self, policy_parameters):
        if self.discrete:
            sy_logits_na = policy_parameters
            
            # tf.multinomial: Draws samples from a multinomial distribution
            sy_sampled_ac = tf.multinomial(logits=sy_logits_na, num_samples=1,name='sample_action')
            
            # convert the shape from (batch_size, 1) -> (batch_size,) = turn [1] in 1
            sy_sampled_ac = tf.squeeze(sy_sampled_ac, [1])
        else:
            sy_mean, sy_logstd = policy_parameters
            
            # get std from log_std
            sy_std = tf.exp(sy_logstd) 
            sy_z_sampled = tf.random_normal(tf.shape(sy_mean),mean=0.0, stddev=1.0,name='z') 
            
            # Using the reparameterization trick
            sy_sampled_ac = tf.add(sy_mean, sy_std * sy_z_sampled) 
        return sy_sampled_ac

    def get_action(self, obs):
        action = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: obs.reshape(1, -1)})
        
        #return as [1], we need to remove the []
        return action[0]


    # Add gaussian noise to the network parameters
    def mutate(self):
        
        # hyper-parameter. Conceptually similar to learning rate
        mutation_power = 0.02 

        # Get list of weights
        variables = tf.global_variables()
        for i in range(len(variables)):
            # Create a vector of random noise with the same shape as each variables and add to it
            variables[i].assign_add(tf.random_normal(tf.shape(variables[i]),mean=0.0,stddev=1.0,dtype=tf.float32) * mutation_power)
            
