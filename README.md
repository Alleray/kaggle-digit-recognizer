MNIST database is a large database of handwritten digits and it's kind of "Hello World" in Machine Learning.
Link to the competition: https://www.kaggle.com/c/digit-recognizer

My model is definitely not the best model there is, but it achieved the accuracy of 0.94885 on the test set.
update: trained model.py for 100 epochs and got a minor improvement in accuracy: submission scored 0.95657, which is an improvement of  previous score of 0.94885.
I will update the model soon, currently planned:
- reshape the input data from a 1D vector back to a 2D matrix.
I think the accuracy should increase because reshaping will bring back the shapes of the digits. (done, model_v2)
Comparing to the previous model, model_v2 takes much more time to train,
more computationally expensive and memory-hungry.
For example, model.py training over 40 epochs with 512 mini-batch size
took ~20 sec on my GTX 650. 20 epochs on model_v2 with 100 mini-batch size took 33 minutes.
Played with hyperparameters and achieved this results:
Loss = 0.2872409
Dev set accuracy: 0.92
[Finished in 1976.5s]
Overall decrease in performance.
So, I think it's not worth to spend more time on this version.
Change the mini-batch size if you are experiencing ResourceExhaustError.

- implement a Convolutional NN architecture. CNNs are better in the field of computer vision. (done, model_cnn) Your submission scored 0.97814, which is an improvement of your previous score of 0.95657. Added: TensorBoard and Checkpoints. Next stop: 99%+ accuracy. Wanna try data augmentation to increase the size of the training set.
- look for the open submissions of other competitors. Should give some useful insights.
