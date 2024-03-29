# FoodGan: comparing ResNet and U-Net
This repository is created to show results of using ResNet and U-Net architechture in CycleGan. In the results section you can see the continious process of model training and difference between these two architechtures. 
# Training
For this project we took pictures of pasta and ghotic architecture. Our goal was to find out differences between two architechtures of CycleGan model: ResNet and U-net. Dataset contains 300 pictures of both. The model was trained for 250 epochs. It might seem that there isn't sufficient time for model to train but even with this number of epochs some conclusions can be made.
# CycleGan model
For the model itself we took basic main parameters:
- Learning rate of 0.0001
- Validation lambda of 1
- Reconstructor lambda of 10
- And for both discriminator and generator filters of 64
Also images were scaled to 256x256 dimension.Increasing filters won't give better results, just decreasing the amount of running time for each epoch. 
Changing the lambdas won't do any better either. If we decrease the learning rate, the discriminator will have a higher loss, what will result in the following:
![Новая таблица - Лист1-2_page-0001](https://user-images.githubusercontent.com/73700350/185763068-920bd576-7702-4b80-b2e2-cc1fc5eca22d.jpg)
# Results
As for results, CycleGan based on U-net architechture tends to produce better result, making building to have pasta structure. Whereas ResNet architechture shows better outcomes at transofrming pasta into building. Thus, it might be concluded that for initial purposes (turning building into pasta structure) U-net is more suitable.
![Новая таблица - Лист1-1_page-0001](https://user-images.githubusercontent.com/73700350/185762630-b587a9bf-8ce4-405f-8a04-3bf8056d8ef1.jpg)
# References
- CycleGan model was taken from David Foster "Generative Deep Learning".
- Pictures of building were taken from Kaggle architechture dataset https://www.kaggle.com/datasets/wwymak/architecture-dataset
- Pictures of pasta were taken from Food 101 dataset testset https://www.kaggle.com/datasets/dansbecker/food-101
- Models, pictures, configurations, and weights are provided in the repository. You can download it to continue training for better results.
