# Word-VAE-experiment

Python scripts for human memory experiment creation. 

Does the following: 
- Makes stimuli for human experiment
- Models human memory with beta-VAE 
- Allows statistical model evaluation and prediction making

### Overview

Human memory is modeled here as lossy compression with a generative model, beta-VAE. It is an unsupervised learning algorithm, with a given beta parameter as compression rate. The more time has passed, and more forgetting happened as in the case of humans; and the higher the beta value in the model, which puts the emphasis on prior knowledge. The more we forget, the more we put the emphasis on our general knowledge.
The human experiment is in the domain of language with 3 kinds of stimuli:
- *original* words: 3rd order english-like words which are shown to the subjects in the human **training phase**
- *VAE* words: original words are distorted with the beta-VAE, these are shown in the **test phase**
- *filler* words: original words are distorted randomly, shown in the **test phase**

After the **training phase** a given time has passed. Hypothesis: In the **test phase**, where people have to choose the already seen letter sequenes, they tend to 'VAE' options, because they are distorted structurally as opposed to randomly. 

The model was trained on an english language corpus. 
