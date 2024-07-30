HI, Welcome to my Project Repository. This will include some simple guidance to help you navigate through my projects.

Although This Repository has a Single Project but there are several Versions of the Project with Different Machine Learning Models and Workaround.

SO Let me just Briefly give you the Project Desciption

**# TOPIC**
-
Predicting the Outcome of Charpy Impact Test on Steel Using Machine Learning Techniques

    Description:

    In this Live Project, we used AI and ML models to predict Charpy Impact Test outcomes, aiming to overcome the limitations of traditional methods such as time-consuming processes and susceptibility to uncertainties. By leveraging computational power and large datasets provided by SAIL's RDCIS department, I developed a robust predictive model that can accurately forecast test results. Through extensive experimentation with various machine learning techniques, I was able to significantly reduce testing time from weeks or months to mere minutes, ultimately establishing a new industry standard for predictive accuracy.

**# PROJECTS**
-

1.	With the Help of Logistics Regression

    ->As the name suggests that this is the Approach to the Project where we make use of Logistics Regression to Find the Maximum capable Result.|

    -> After playing around with it for 2 days I finally reached at the Optimal result that this method was able to provide i.e. the Maximum Possible Accuracy of the Model
                                                                                
2.	With the Help of Support Vector Machine

    -> As the name suggests that this is the Approach to the Project where we make use of Support Vector Machine to Find the Maximum capable Result.

    -> After playing around with it for 1 day I finally reached at the Optimal result that this method was able to provide i.e. the Maximum Possible Accuracy of the Model.


3.	With the Help of K-Nearest Neighbours

    -> As the name suggests that this is the Approach to the Project where we make use of KNN to Find the Maximum capable Result.
    
    -> After playing around with it for 3 days I finally reached at the Optimal result that this method was able to provide i.e. the Maximum Possible Accuracy of the Model.


4.	With the Help of Artificial Neural Network

    -> As the name suggests that this is the Approach to the Project where we make use of ANN to Find the Maximum capable Result.

    -> This is the only one which stood out to be slightly a bit more complex than the other two. As It had way too many parameters to adjust, change and play around alongside studying what each of the parameters mean and how they’d affect the overall program along with the result.

    -> After adjusting and refining it for 2 weeks I finally reached at the Optimal result that this method was able to provide i.e. the Maximum Possible Accuracy of the Model


Some Common Steps followed in each of the Cases

    •	Dividing the entire data into 75% training dataset and 25% testing dataset.
    •	Using Panda Library to Extract the Rows and Columns from the Given Data
    •	Calculating and Plotting these Results - Accuracy, Precision, Recall score, F1 score, Area Under the Precision-Recall Curve (AUC PR), Receiver Operating Characteristic - Area Under the Curve (ROC AUC), Root Mean Square Error (RMSE).
    •	Saved each of these metrices for Further Comparison between each of them.


**# RESULTS**

    •   The results for the three given models can be summarized as:

    MODEL                       RMSE	F1      	ROC AUC	    AUC PR
    -------------------------------------------------------------------
    Logistics Regression	    13.74	99.05	    50	        99.37
    SVM Regressor	            13.74	99.05	    50	        98.92
    KNN Regressor	            13.74	99.05	    50	        99.1
    Artificial Neural Network	13.74	99.04	    98.32	    99.97

    •	We are dealing with a data set whose Output is in binary format
    •	As we can see the 4 models have nearly the same output values except the " Artificial-Neural Network ".
    •	This proves that a majority is being formed during the testing of the dataset
    •	Therefore the most accurate value that we should look at in order to draw a conclusion are the ‘ROC-AUC’ and ‘AUC-Pr’.
    •	Higher the ‘AUC-Pr’ and ‘ROC-AUC’ with Lower or same level of RMSE the better is the performance of a model during a majority occurrence.
    •	Thus we can easily draw the  conclusion that “Artificial Neural Network” or we can also say that “Deep Learning” is the best model while working with a ‘Binary’ dataset.


Also it should be noted that NONE of them are a ‘Bad’ model for this type of data

    •Because they each had their own Pros and Cons:-

        -   LR/SVM/KNN (Machine Learning) :- Provides Lower Accurate results but with Higher Performance and Speed

        -   ANN (Deep Leaning) :- Provides Highly Accurate results but with the cost of Lower Performance and Speed and also more Training Time than the other.
