#!/usr/bin/env python
# coding: utf-8
import os
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np
from traindata_generation import *
from stimuli_generation import *


def next_batch(data, batch_size):
    """
    Chooses random batch.
    """
    indices = np.random.randint(0, len(data), batch_size, dtype=int)
    batch = data[indices]
    return batch

class BetaVAE:
    """
    Class for beta-VAE model. 
    Recieves words with specified length 'word_length'. Can be single or multilayer with already specified size. 
    Encoder and Decoder can be linear or nonlinear. Latent space dimension size is given by 'z_size'.
    """
    def __init__(self, learning_rate=1e-4, word_length=7, z_size=20, layer="single", encoder="nonlinear", decoder="nonlinear"):
        self.learning_rate = learning_rate
        self.word_length = word_length
        self.z_size = z_size
        self.layer = layer
        self.encoder = encoder
        self.decoder = decoder
        self.train_length = 0

        if self.layer == "single":
            self.l_enc = [400]
            self.l_dec = [400]
            print("SLP")
        elif self.layer == "multi":
            self.l_enc = [512, 256]
            self.l_dec = [256, 512]
            print("MLP")
        
        tf.disable_eager_execution()

    def blocks(self, var, current_size, levels, ltype, coder):
        """
        Helper function for buliding encoder/decoder hidden layers.
        """
        for idx, level in enumerate(levels):
            with tf.variable_scope(ltype + '_layer' + str(idx)):

                weights = tf.get_variable(ltype + 'w' + str(idx), [current_size, level])
                bias = tf.get_variable(ltype + "b" + str(idx), [level])
                
                if coder == "nonlinear":
                    var = tf.nn.relu(tf.add(tf.matmul(var, weights), bias))
                    print("nonlinear")
                elif coder == "linear":
                    var = tf.add(tf.matmul(var, weights), bias)
                    print("linear")

                current_size = level
        return current_size, var

    def create_encoder(self):
        """
        Creates encoder. Sets the parameters of the Gaussian posterior distribution.
        """
        current_size = self.word_length * LEN_ABC
        x = tf.layers.flatten(self.x_input)
        current_size, x = self.blocks(x, current_size, self.l_enc, "Encoder", self.encoder)

        w_mu = tf.get_variable("w_mu", [current_size, self.z_size])
        b_mu = tf.get_variable("b_mu", [self.z_size])

        w_sigma = tf.get_variable("w_sigma", [current_size, self.z_size])
        b_sigma = tf.get_variable("b_sigma", [self.z_size])
        
        self.z_mu = tf.add(tf.matmul(x, w_mu), b_mu)
        self.z_sigma = tf.exp(tf.add(tf.matmul(x, w_sigma), b_sigma))
    
    def create_decoder(self):
        """
        Creates decoder. Gives back the Categroical output distribution.
        """
        current_size = self.z_size
        x_hat_size = self.word_length * LEN_ABC

        current_size, z = self.blocks(self.z_sample, current_size, self.l_dec, "Decoder", self.decoder)

        w = tf.get_variable("w_m", [current_size, x_hat_size])
        b = tf.get_variable("b_m", x_hat_size)     

        self.logits = tf.add(tf.matmul(z, w), b)

        self.logits = tf.reshape(self.logits, (-1, self.word_length, LEN_ABC))
        self.x_hat = tf.nn.softmax(self.logits, axis=-1)

    def create_graph(self, sample="posterior", sample_size=None):
        """
        Model generation.
        """
        tf.reset_default_graph()
        print("Creating graph...")
        input_size = [self.word_length, LEN_ABC]
        batch_size = None # dynamical batch size
        
        # Sampling is from posterior distribution by default, during training and reconstruction as well.
        # Therefore an Encoder is created.
        if sample == "posterior":
            self.x_input = tf.placeholder(tf.float32, [batch_size] + input_size)

            self.create_encoder()

            self.q_post = tfp.distributions.MultivariateNormalDiag(loc=self.z_mu, scale_diag=self.z_sigma)
            
            # Sampling from Posterior.
            self.z_sample = self.z_mu + tf.random.normal(tf.shape(self.z_sigma)) * self.z_sigma
        
        # For model examination sampling from Prior distribution is also an option. 
        elif sample == "prior":
            print("Sampling from Prior!")
            self.z_sample = tf.random.normal((sample_size, self.z_size))
            self.create_decoder()

            # likelihood
            self.p_likelihood = tfp.distributions.OneHotCategorical(logits=self.logits) 

            # Probability under the model
            self.x_distorted = tf.placeholder(tf.float32, [batch_size] + input_size)
            P_all = self.p_likelihood.log_prob(self.x_distorted)
            self.P = tf.reduce_sum(P_all, axis = 1)
            return self.x_distorted, self.x_hat, self.P

        self.create_decoder()

        # prior
        self.prior_sigma = np.full((self.z_size), 1.0, dtype=np.float32)
        self.prior_mu = np.full((self.z_size), 0.0, dtype=np.float32)
        self.p_prior = tfp.distributions.MultivariateNormalDiag(loc=self.prior_mu, scale_diag=self.prior_sigma)
        
        # likelihood
        self.p_likelihood = tfp.distributions.OneHotCategorical(logits=self.logits) 

        # Probability under the model
        self.x_distorted = tf.placeholder(tf.float32, [batch_size] + input_size)
        P_all = self.p_likelihood.log_prob(self.x_distorted)
        self.P = tf.reduce_sum(P_all, axis = 1)

        # Losses and beta for training
        self.beta_in = tf.placeholder(tf.float32)
        self.Rate = tf.reduce_sum(-tfp.distributions.kl_divergence(self.q_post, self.p_prior))  
        self.Dist = tf.reduce_sum(self.p_likelihood.log_prob(self.x_input)) 
        self.loss = -(self.beta_in * self.Rate + self.Dist)

        # Model accuracy
        self.acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.math.equal(tf.math.argmax(self.x_input,axis = 2),tf.math.argmax(self.x_hat,axis = 2)), tf.float32), axis=1)/self.word_length)
        self.correct_num = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.math.equal(tf.math.argmax(self.x_input,axis = 2),tf.math.argmax(self.x_hat,axis = 2)), tf.float32), axis=1))

        # Model optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        return self.beta_in, self.x_input, self.x_hat, self.z_mu, self.z_sigma, self.z_sample, self.x_distorted, self.optimizer, self.Rate, self.Dist, self.acc, self.correct_num, self.P

    def train(self, model_name, model_dir, train_data, train_length, acc_verbose=True, test_data=[], target_beta=0.3):
        """
        Model training. 
        Optionally gives back training curve with Reconstrution- and KL-loss, and test set accuracy. Accuracy measured in erronous letters / sequence.
        Training is done with tempered beta (beta increases with training to a given target value).
        """                
        self.train_length = train_length
        try:
            input = np.load(train_data)
        except:
            print("Training data npy file doesn't exit!")
            exit()

        input = np.reshape(input, (input.shape[0], self.word_length, LEN_ABC))
        print("Training file is loaded.")
        if test_data:
            try:
                input = np.load(test_data)
            except:
                print("Test data npy file doesn't exit!")
                exit()
            test = np.reshape(test, (test.shape[0], self.word_length, LEN_ABC))
            print("Test file is loaded.")

        beta_fraction = target_beta/(self.train_length/4) # tempered beta

        self.beta_in, self.x_input, _, _, _, _, _, self.optimizer, self.Rate, self.Dist, self.acc, self.correct_num, _ = self.create_graph()
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=1)

            beta = beta_fraction
            acc_plot = []
            kl_plot = []
            likel_plot = []
            acc_plot_test = []

            for epoch in range(self.train_length):
                input_data = next_batch(input, 100)
                
                if acc_verbose and epoch % (self.train_length / 100) == 0:
                    _, kl, likel, a, correct = sess.run([self.optimizer, self.Rate, self.Dist, self.acc, self.correct_num],
                                             feed_dict={self.x_input: input_data, self.beta_in: beta})
                    print(epoch, ". optim: " + model_name)
                    print(" \t Losses (kl, likelihood): ", kl, likel, ", Acc: ", a * 100, "% , MEAN ERROR/SEQUENCE: ", self.word_length - correct)
                    acc_plot.append(a * 100)
                    kl_plot.append(-kl / 10)
                    likel_plot.append(-likel)

                    if test_data:
                        test_data = next_batch(test, 1000)
                        a_test = sess.run(self.acc, feed_dict={self.x_input: test_data})
                        acc_plot_test.append(a_test * 100)
                else:
                    _ =  sess.run(self.optimizer, feed_dict={self.x_input: input_data, self.beta_in: beta})

                if beta < target_beta:
                    beta += beta_fraction     
                    
            saver.save(sess, os.path.join(model_dir, model_name + "_len" +str(self.word_length) + "_" + str(self.z_size) + "D_" + self.layer + "_enc" + self.encoder + "_dec" + self.decoder), global_step=epoch)    
        
        if acc_verbose:
            return acc_plot, kl_plot, likel_plot , acc_plot_test
        else:
            return


    def posterior_retrieve(self, model_dir, model_name, train_length, input_txt):
        """
        Gives back the posterior distribution parameters of given input under an already trained model. 
        Output: ([z_mu, z_sigma], words, dimensions)
        """
        onehot_input = text_to_onehot(loading(input_txt).rsplit(), self.word_length)
        _, self.x_input, _, self.z_mu, self.z_sigma, _, _, _, _, _, _, _, _ = self.create_graph()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_dir, model_name + "_len" +str(self.word_length) + "_" + str(self.z_size) + "D_" + self.layer + "_enc" + self.encoder + "_dec" + self.decoder + "-" + str(train_length-1)))
            posterior = sess.run([self.z_mu, self.z_sigma], feed_dict={self.x_input: onehot_input})

        return posterior


    def likelihood_retrieve(self, model_dir, model_name, train_length, input_txt):
        """
        Gievs back likelihood Categorical distribution of given input under an already trained model.
        Output: (words, letter places, probability of each english letter)
        """
        onehot_input = text_to_onehot(loading(input_txt).rsplit(), self.word_length)
        _, self.x_input, self.x_hat, _, _, _, _, _, _, _, _, _, _ = self.create_graph()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_dir, model_name + "_len" + str(self.word_length) + "_" + str(self.z_size) + "D_" + self.layer + "_enc" + self.encoder + "_dec" + self.decoder + "-" + str(train_length-1)))
            categorical = sess.run(self.x_hat, feed_dict={self.x_input: onehot_input})

        return categorical


    def prior_sampler(self, model_dir, model_name, train_length, sample_size=100):
        """
        Sampling from prior distribution under an already trained model.
        Output: (100 sample, letter places, probability of each english letter)
        """
        _, self.x_hat, _ = self.create_graph(sample="prior", sample_size=sample_size)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_dir, model_name + "_len" + str(self.word_length) + "_" + str(self.z_size) + "D_" + self.layer + "_enc" + self.encoder + "_dec" + self.decoder + "-" + str(train_length-1)))
            categorical = sess.run(self.x_hat, feed_dict={})
        
        return categorical


    def reconstruction(self, model_dir, model_name, train_length, input_txt, output_txt_name):
        """
        Gives back the list of reconstructed words by a given model of given 'input_txt' and the average error rate (erronous letter/sequence). 
        Saves output list in txt file.
        """
        input_txt = loading(input_txt).rsplit()
        onehot_input = text_to_onehot(input_txt, self.word_length)     
         
        _, self.x_input, self.x_hat, _, _, _, _, _, _, _, _, self.correct_num, _ = self.create_graph()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_dir, model_name + "_len" + str(self.word_length) + "_" + str(self.z_size) + "D_" + self.layer + "_enc" + self.encoder + "_dec" + self.decoder + "-" + str(train_length-1)))
            onehot_output, corr = sess.run([self.x_hat, self.correct_num], feed_dict={self.x_input: onehot_input})
        
        vae_words = []
        with open(output_txt_name, "w+")as file:
            for word in onehot_output:
                word = onehot_to_word(word)
                file.write(word + "\n")
                vae_words.append(word)
        average_error = np.mean(self.word_length - corr)
        
        return vae_words, average_error

    def reconstruction_probability(self, model_dir, model_name, train_length, input_txt, output_txt, sample_size=1000):
        """
        Gives back probability of reconstructing 'output_txt' given 'input_txt'. P(x_hat|x, model)
        Sampling is needed for better results, default is 1000 samples' average. 
        """
        input = loading(input_txt).split()
        onehot_input = text_to_onehot(input, self.word_length)
        onehot_output = text_to_onehot(loading(output_txt).split(), self.word_length)

        _, self.x_input, _, self.z_mu, self.z_sigma, _, self.x_distorted, _, _, _, _, _, self.P = self.create_graph()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_dir, model_name + "_len" + str(self.word_length) + "_" + str(self.z_size) + "D_" + self.layer + "_enc" + self.encoder + "_dec" + self.decoder + "-" + str(train_length-1)))
            post_mu, post_sigma  = sess.run([self.z_mu, self.z_sigma],feed_dict = {self.x_input: onehot_input})
            probs = np.zeros((sample_size, len(input)))
            for i in range(sample_size):
                probs[i] = sess.run(self.P,feed_dict = {self.z_mu: post_mu, self.z_sigma: post_sigma, self.x_distorted: onehot_output})

        return np.mean(np.exp(probs), axis=0)

    def generative_probability(self, model_dir, model_name, train_length, output_txt, sample_size=1000):
        """
        Gives back probability of generating 'output_txt' under model. P(x_hat|model)
        Sampling is needed for better results, default is 1000 samples' average. 
        """
        output = loading(output_txt).split()
        onehot_output = text_to_onehot(output, self.word_length)
        self.x_distorted, self.x_hat, self.P = self.create_graph(sample="prior", sample_size=len(output))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_dir, model_name + "_len" + str(self.word_length) + "_" + str(self.z_size) + "D_" + self.layer + "_enc" + self.encoder + "_dec" + self.decoder + "-" + str(train_length-1)))
            probs = np.zeros((sample_size, len(output)))
            for i in range(sample_size):
                probs[i] = sess.run(self.P,feed_dict = {self.x_distorted: onehot_output})

        return np.mean(np.exp(probs), axis=0)


