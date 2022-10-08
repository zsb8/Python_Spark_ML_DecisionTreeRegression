Data is pubic. We want to use ML to predict the number of bicycle rentals. 

# Python_Spark_ML_RMSE
Use RMSE (Root Mean Square Error) to find evergreen webside. 

Running environment is Spark + Hadoop + PySpark    
Used the algorithm is DecisionTree regession.     
Used the library is pyspark.mllib and RegessionMetrics. 

# Stage1:  Read data
Placed the hour.csv on hadoop. Built 3 data sets: (1) Train data, (2) Validation data, (3) Sub_test data.

## Compare the parameters
"maxDepth"
Set the impurity='variance' and bins=50, draw the graph for the numIterations. The RMSE is the highest when depth=3. 
~~~
    impurity_list = ["variance"]
    max_depth_list = [3, 5, 10, 15, 20, 25]
    max_bins_list = [50]
~~~
![image](https://user-images.githubusercontent.com/75282285/194718627-fef4af6d-8bc6-4867-8049-f0a535fc1887.png)


"maxBins"
Set the impurity='variance' and depth=3, draw the graph for the numIterations. The RMSE is the highest when bins=5. 
~~~
    impurity_list = ["variance"]
    max_depth_list = [3]
    max_bins_list = [3, 5, 10, 50, 100, 200]
~~~
![image](https://user-images.githubusercontent.com/75282285/194719315-f3a46599-f1af-48b2-96ed-0a12381c3725.png)



# Stage2: Train and evaluate   
Created the model using train data set.   
Calculated the RMSE using validation data set.
Sorted the metrics.    
Found the best parameters includ the best accuracy and the best model.  
~~~
def train_evaluation_model(train_data,
                           validation_data,
                           impurity,
                           max_depth,
                           max_bins):
    start_time = time()
    model = DecisionTree.trainRegressor(train_data,
                                         categoricalFeaturesInfo={},
                                         impurity=impurity,
                                         maxDepth=max_depth,
                                         maxBins=max_bins
                                         )
    accuracy = evaluate_model(model, validation_data)
    duration = time() - start_time
    return accuracy, impurity, max_depth, max_bins, duration, model
~~~
![image](https://user-images.githubusercontent.com/75282285/194719343-cd54ec15-168c-4abc-b6cb-3962250d4cfb.png)



# Stage3: Test
Used the sub_test data set and the best model to calculate the RMSE. If testing RMSE is similare as the best RMSE, it is OK.
As the result, the best RMSE is 143.2841, use the test data set to calcuate RMSE is 139.0207, the difference is 4.2634, so it has not overfitting issue. 
![image](https://user-images.githubusercontent.com/75282285/194720733-07f3ad85-968d-4221-9d0c-5d8836ea7a15.png)


# Stage4: Predict
~~~
def predict_data(best_model):
    for lp in label_point_RDD.take(20):
        predict = best_model.predict(lp.features)
        label = lp.label
        features = lp.features
        result = ("Correct" if  (label == predict) else "Error")
        print("Forest：Elevation" + str(features[0]) +
                 " Aspect:" + str(features[1]) +
                 " Slope:" + str(features[2]) +
                 " Vertical_Distance_To_Hydrology :" + str(features[3]) +
                 " Horizontal_Distance_To_Hydrology:" + str(features[4]) +
                 " Hillshade_9am :" + str(features[5]) +
                 "....==>Predict:" + str(predict) +
                 " Fact:" + str(label) + "Result:" + result)
~~~
![image](https://user-images.githubusercontent.com/75282285/194675371-c2aa861c-9f4f-444b-9da4-1eccea269a02.png)


# Spark monitor
http://node1:8080/    
![image](https://user-images.githubusercontent.com/75282285/194720513-4badd7c2-ae22-4c67-9eba-77dccae28bb1.png)

http://node1:4040/jobs/   
![image](https://user-images.githubusercontent.com/75282285/194720490-042ba697-a6c7-49bd-ba29-fc500e182444.png)

