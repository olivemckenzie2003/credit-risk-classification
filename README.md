# credit-risk-classification
https://github.com/olivemckenzie2003/credit-risk-classification/files/11057511/ReadMe.for.challenge.20.homework.docx

Report
Overview of the Analysis
The purpose of this analysis is to predict from a given data set of 77537 records if a loan was healthy or of high risk. 75036 records were safe loans and 2500 were high risk loans.  The data set contained eight columns which were as follows: 
loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt and loan status.

Various techniques were used to train and evaluate the dataset of a historical lending activity from a peer-to-peer lending services company.
The analysis completed for the machine learning models used in this Challenge were as follows:

1.	Split the data set into training and test data set with the following process
 
a.	Read lending.csv file from resources folder into Pandas DataFrame

b.	Create label set by initialising  y = “loan_status” (This column consists of “0” and “1” values where 0=healthy loan and 1=high risk loan)
c. Create features set by initialising X=to the remainder of the DataFrame
    columns

d. Check the balance of the variable which is the same as counting how many
   1s and 0s are in the column.

e. Using the sklearn. Model_selection import train_test_split library to produce 
    X-train, X_test,y_train, and y_test

Code:
#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)


2.Create a Logistic Regression Model with the Original Data
 	a. Fit the logistic regression model by using training data (X_train and y-train)
		1. import the LogisticRegression module from sklearn.linear_model
		2. Instantiate the Logistic Regression model
		3. Fit model using the training data X-train and y_train

b.	Save the predictions on the data labels by using the testing feature data (X_test) and the fitted model.

b1. Make a prediction using test data X_test

3.Evaluate the model’s performance by doing the following:
		a. Calculate the accuracy score of the model.
a1. Print the balance accuracy score using the y_testing and testing 
     predictions

		b. Generate confusion matrix.
		c. Print the classification report.
4. Predict a Logistic Regression Model with Resampled training Data

a.	Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points, no they do not. There is not an even number of 0s and 1s

b.	Fit the original training data(X_train and y_train) to the random_overSampler model
c.	Count the distinct values of the resampled labels and data

Get even value of 0s and 1s: thus a balanced data set
0    56277
1    56277

5. Use the Logistic Regression classifier and the resampled data to fit the model and make
    Predictions

a.	Instantiate the logistic Regression Model.

b.	Fit the model using the resampled training data. (X_res, y_res)

c.	Make predictions using the testing data.

6.    Evaluate the models performance by:

	a.  Calculate the accuracy score of the model.
          a1. Print the balance accuracy score using the y_testing and testing
              predictions

	b. Generate confusion matrix.
	c. Print the classification report.

 

The Confusion Matrix

The confusion matrix is a matrix of numbers which shows where the 
model becomes confused. It maps the predictions to the original classes to which the data belong. It is a class-wise distribution of the predictive 
performance of a classification model. It can only be used in supervised 
learning where the output distribution is known and computation is done on one test set of a dataset using different classifiers, which allow for 
comparison of their relative strengths and weaknesses and shows how 
they can be combined for optimal performance. Reference Google

Confusion matrix for this binary class classification problems classified as follows:
•	True Positive (TP) refers to a sample belonging to the positive class being classified correctly.
•	True Negative (TN) refers to a sample belonging to the negative class being classified correctly.
•	False Positive (FP) refers to a sample belonging to the negative class but being classified wrongly as belonging to the positive class.
•	False Negative (FN) refers to a sample belonging to the positive class but being classified wrongly as belonging to the negative class.
 
Ref: Google

In this challenge with a loan status of heathy and high risk loans 0 and 1 respectively and is classified as a binary dataset with two distinct categories of data, which can be named “positive” and “negative”. This data set is also classed as an imbalanced data set because it has an unequal number of 0s and 1s. The function y.value_counts counts the number of items in each data set. In this data set we have 75036 of 0s and 2500 of 1s which will be used to evaluate the learning model.
0    75036
1     2500



The train_test_split function takes a sample of data from the main data set to use in the confusion function matrix.
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)


Output
Logistic Regression Model with the Original Data Output:
Confusion Matrix
		Predicted 0	Predicted 1
Actual 0	18679		80
Actual 1	67		558

Total Heathy Loans =18679+80=18759
High Risk Loans=67+558=625

Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     18759
           1       0.87      0.89      0.88       625

    accuracy                           0.99     19384
   macro avg       0.94      0.94      0.94     19384
weighted avg       0.99      0.99      0.99     19384


Logistic Regression Model with Resampled training Data Output
Confusion Matrix
		Predicted 0	Predicted 1
   Actual 0	   18668		91                   
   Actual 1		2		623
		
Total Heathy Loans =18668+91=18759
High Risk Loans=2+623=625

Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     18759
           1       0.87      1.00      0.93       625

    accuracy                           1.00     19384
   macro avg       0.94      1.00      0.96     19384
weighted avg       1.00      1.00      1.00     19384


Machine Learning Model 1: Logistic Regression Model with the Original Data
Precision: 100% predictions of healthy loans were correct
 	      87% predictions of high-risk loans were correct.
	      94% predictions of micro Avg were correct.
	      99% predictions of weighted avg were correct.

Recall: For healthy loans the percentage of positive cases caught were 100%
	  For high risk loans the percentage of positive cases caught were 89%
        For micro Avg loans the percentage of positive cases caught were 94%
	  For weighted avg loans the percentage of positive cases caught were 99%

F1 Score: Percentage of positive correct predictions for healthy loans were 100%
	    Percentage of positive correct predictions for high risk loans were 88%
	    Percentage of positive correct predictions for accuracy was 99%
          Percentage of positive correct predictions for micro Avg loans was 94%
          Percentage of positive correct predictions for weighted Avg loans was
          99%
	

Support: 18759 heathy loan records used
     	   625 high risk records used.
         19384 total heathy loan records and high risk records used for accuracy.
         19384 total heathy loan records and high risk records used for macro avg.
         19384 total heathy loan records and high risk records used for weighted avg.


Machine Learning Model 2: Logistic Regression Model with Resampled training Data
Precision: 100% predictions of healthy loans were correct
 	     87% predictions of high-risk loans were correct.
	     94% predictions of micro Avg were correct.
	     100% predictions of weighted avg were correct.

Recall: For healthy loans the percentage of positive cases caught were 100%
	  For high risk loans the percentage of positive cases caught were 100%
        For micro Avg loans the percentage of positive cases caught were 100%
	  For weighted avg loans the percentage of positive cases caught were 100%

F1 Score: Percentage of positive correct predictions for healthy loans were 100%
	    Percentage of positive correct predictions for high risk loans were 93%
	    Percentage of positive correct predictions for accuracy was 100%
          Percentage of positive correct predictions for micro Avg loans was 96%
          Percentage of positive correct predictions for weighted Avg loans was
          100%
	

Support: 18759 heathy loan records used
         625 high risk records used.
         19384 total heathy loan records and high risk records used for accuracy.
         19384 total heathy loan records and high risk records used for macro avg.
         19384 total heathy loan records and high risk records used for weighted avg.






Conclusion 

The Logistic Regression Model with Resampled training Data Machine Learning 
model produces the best results where recall where the number of positive cases caught were 100%, therefore 100% of all risky loans in the data set was flagged up. Whereas in the Logistic Regression Model with the Original Data 11% of high risk loans were missed. This could mean high losses for a loan company depending on the size of the loans.

For precision there was no difference between the two models. However, with f1-score there was a slight improvement 5% of positive correct predictions for high risk loans using the Logistic Regression Model with Resampled training Data Machine Learning model. Which could save the loan company a lot of money depending on the size of the loan.

Overall prediction accuracy Logistic Regression Model with Resampled training Data Machine Learning model for f1-score was 1% better than the Logistic Regression Model with the Original Data.


On the bases this analysis the The Logistic Regression Model with Resampled training Data Machine Learning is the best model to choose to process this loan company's data with.



















