MNIST database is a large database of handwritten digits and it's kind of "Hello World" in Machine Learning.
Link to the competition: https://www.kaggle.com/c/digit-recognizer

My model is definitely not the best model there is, but it achieved the accuracy of 0.94885 on the test set.
I will update the model soon, currently planned:
- reshape the input data from a 1D vector back to a 2D matrix.
I think the accuracy should increase because reshaping will bring back the shapes of the digits. (done, model_v2)
Comparing to the previous model, model_v2 takes much more time to train,
more computationally expensive and memory-hungry.
For example, model.py training over 40 epochs with 512 mini-batch size
took ~20 sec on my GTX 650. 20 epochs on this model with 100 mini-batch size took 33 minutes.
Played with hyperparameters and achieved this results:
Loss = 0.2872409
Dev set accuracy: 0.92
[Finished in 1976.5s]
Overall decrease in performance.
So, I think it's not worth to spend more time on this version.
Change the mini-batch size if you are experiencing ResourceExhaustError.

- implement a Convolutional NN architecture. CNNs are better in the field of computer vision.
- look for the open submissions of other competitors. Should give some useful insights.
