# FoodGan: comparing ResNet and U-net
This repository is created to show results of using ResNet and U-net architechture in CycleGan. 

# Training
For this project we took pictures of pasta and ghotic architecture. Our goal was to find out differences between two architechtures of CycleGan model: ResNet and U-net. Dataset contains 300 pictures of both. The model was trained for 250 epochs. It might seem that there isn't sufficient time for model to train but even with this number of epochs some conclusions can be made.
# CycleGan model
For the model itself we took basic main parameters:
- Learning rate of 0.0001
- Validation lambda of 1
- Reconstructor lambda of 10
- And for both discriminator and generator filters of 64
Also images were scaled to 256x256 dimension.
Increasing filters won't give better results, just increasing the amount of running time for each epoch. 
Changing the lambdas won't do any better either. If we decrease the learning rate, the discriminator will have a higher loss, what will result in the following:
# Results
As for results, ... check

