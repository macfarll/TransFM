This is a work-in-progress rework of the TransFM code here: https://github.com/rpasricha/TransFM

More feature demos and documentation will be added soon.

While the original code was from the above repository, the changes here are extensive and include:

-Changing Tensorflow version to 2.x
-Adding functionality for outputting predections
-Memory Optimizations most notably the ability to train in segments
-Adding option to use regional data, limiting users to interacting with only certain items
-Ability to save and load model in a way that preserves user data
-Ability to add new items and users to a partially trained model 
-Option to only use train/test, allowing more users to be covered by model
-Model seeding allows for determininsitic results
-Args object allows for easier debugging and saving of run parameters 
-Added pyton notebook that can be used in google colab to quickly demo model