### Learned NDV estimator
Learned model to estimate number of distinct values (NDV) of a population using a small sample. The model approximates the maximum likelihood estimation of NDV, which is difficult to obtain analytically.

### How to use
1. Create an instance
   
    `estimator = Estimator()`
   
2. Assume your sample is S=[1,1,1,3,5,5,12] and the population size is N=100000. You can estimate population ndv by:

    `ndv = estimator.sample_predict(S=[1,1,1,3,5,5,12], N=100000)`
   
3. If you have the sample profile e.g. f=[2,1,1], you can estimate population NDV by:
   
    `ndv = estimator.profile_predict(f=[2,1,1], N=100000)`

4. If you have multiple samples/profiles from multiple populations, you can estimate population NDV for all of them in a batch by method `estimator.sample_predict_batch()` or `estimator.profile_predict_batch()`.



### How to train the ndv estimator
You can directly use the trained model for your datasets, as the trained model is agnostic to workloads. If you want to train the model from scratch anyway, do the following:
1. Go to the model_training folder
    `cd model_training`

2. Install requirements
   
    `pip install requirements.txt`
   
3. Generate training data. (This uses a lot of memory.)
   
    `python training_data_generation.py`
   
4. Train model
   
    `python model_training.py`
5. Save trained pytorch model parameters to numpy

    `python torch2npy.py`

6. Use the estimate.py script to test your model. Make sure to change `para_path`.