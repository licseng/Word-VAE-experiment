from betaVAE_class import BetaVAE
from stimuli_generation import *
from traindata_generation import *

"""Example code"""


# Saving training data into separate file and training beta-VAE model
training_data_making("proba", ["books_for_model_training/30books_eng.txt"])
data = "proba_train_7length.npy"
bvae = BetaVAE(layer="single")
bvae.train("proba", "models", data, 5000)

# Reconstruction of generated lower order 'orig' words into 'VAE' words
orders_generation("books_for_model_training/15books_eng_for_test.txt", "")
vae_list, error = bvae.reconstruction( "models","proba", 5000, "w7_3.txt", "w7_3_VAE.txt")

# Chosing HD=2 orig and VAE words and making 'filler' words of them
HD_sorting("w7_3.txt", "w7_3_VAE.txt")
filler_making("w7_3_HD2.txt", "w7_3_VAE_HD2.txt", "w7_3_filler_HD2.txt", mode="notVAEplace_1st")

# Cleaning 'orig', 'VAE', 'filler' stimuli to make it ready for experimenting
cleaning("w7_3_HD2.txt", "w7_3_VAE_HD2.txt", "w7_3_filler_HD2.txt")

# Calculating probabilities of stimuli words under given model
probs = bvae.reconstruction_probability("models", "proba", 5000,  "w7_3.txt", "w7_3_VAE.txt")
gen_probs = bvae.generative_probability("models", "proba", 5000,  "w7_3_VAE.txt")

# Retrieving model attributes for model examination and model evaluation
z = bvae.posterior_retrieve("models", "proba", 5000, "w7_3.txt")
cat = bvae.likelihood_retrieve("models", "proba", 5000, "w7_3.txt")
prior_samples = bvae.prior_sampler("models", "proba", 5000)
