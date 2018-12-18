# DIFFICULTIES


### HARDWARE/TIME
As the training process is computationally expensive we opted to train the model 
on Google cloud servers. In spite of the upgrade in hardware, time was still
a major bottleneck as a model on average took 22-24hrs to train.

### COSTS
There were fees associated with the servers and we ended up spending a significant 
portion of our allowance training the models to the desired stage. (~$288 in
total)

### DOCUMENTATION
The TensorFlow CycleGAN API had been partially documented, but most noticably
was missing a critical part of the example reference which detailed how the
models were setup. This meant a lot of time was spent trying to re-implement the
CycleGAN model from scratch with a lot of guesswork as to how to best utilise
the TensorFlow APIs with it.
We opened an issue on the GitHub repository to report this: https://github.com/tensorflow/models/issues/5548
