# credit-risk-classification



Report

Overview of the Analysis


The purpose of this analysis is to predict from a given data set of 77537 records if a loan was healthy or of high risk. 75036 records were safe loans and 2500 were high risk loans.  The data set contained eight columns which were as follows: 
loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt and loan status.

Various techniques were used to train and evaluate the dataset of a historical lending activity from a peer-to-peer lending services company.
The analysis completed for the machine learning models used in this Challenge were as follows:

![split](https://user-images.githubusercontent.com/115945473/227413883-8edf75e2-4cd8-44fc-8113-ab03a306f2b8.jpg)

![TwoLog](https://user-images.githubusercontent.com/115945473/227413910-b7f9ee21-4316-49c0-9d89-b6697090882d.jpg)

![four](https://user-images.githubusercontent.com/115945473/227413961-35ffdabc-4aeb-43ee-8dd2-d58e236a0be2.jpg)
		


The Confusion Matrix
![3_confusion_matrix](https://user-images.githubusercontent.com/115945473/227404757-6615b395-2c10-49d2-b048-e6357b33dff0.png)

The confusion matrix is a matrix of numbers which shows where the 
model becomes confused. It maps the predictions to the original classes to which the data belong. It is a class-wise distribution of the predictive performance of a classification model. It can only be used in supervised learning where the output distribution is known and computation is done on one test set of a dataset using different classifiers, which allow for comparison of their relative strengths and weaknesses and shows how they can be combined for optimal performance. Reference Google

![confusion](https://user-images.githubusercontent.com/115945473/227415034-a9b976a0-9bd1-4397-97ce-d3e2757874b2.jpg)

 
Ref: Google

In this challenge with a loan status of heathy and high risk loans 0 and 1 respectively and is classified as a binary dataset with two distinct categories of data, which can be named “positive” and “negative”. This data set is also classed as an imbalanced data set because it has an unequal number of 0s and 1s. The function y.value_counts counts the number of items in each data set. In this data set we have 75036 of 0s and 2500 of 1s which will be used to evaluate the learning model.



0   - 75036
1   -  2500



The train_test_split function takes a sample of data from the main data set to use in the confusion function matrix.
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=1, 
                                                    stratify=y)


Output

![REGRESSION1](https://user-images.githubusercontent.com/115945473/227406055-ee532259-358c-42e4-a690-02379eb85715.jpg)

![REGRESSION2](https://user-images.githubusercontent.com/115945473/227406073-9fa565af-8a53-4274-9a6b-dbee7048ab6c.jpg)







![ONE](https://user-images.githubusercontent.com/115945473/227406399-e5e54e1b-56ac-436c-8ce4-b40b6fb14e2e.jpg)




![TWO](https://user-images.githubusercontent.com/115945473/227406170-e422d899-241f-4b02-8138-bf1bef9813be.jpg)







Conclusion 


The Logistic Regression Model with Resampled training Data Machine Learning 
model produces the best results where recall where the number of positive cases caught were 100%, therefore 100% of all risky loans in the data set was flagged up. Whereas in the Logistic Regression Model with the Original Data 11% of high risk loans were missed. This could mean high losses for a loan company depending on the size of the loans.


For precision there was no difference between the two models. However, with f1-score there was a slight improvement 5% of positive correct predictions for high risk loans using the Logistic Regression Model with Resampled training Data Machine Learning model. Which could save the loan company a lot of money depending on the size of the loan.


Overall prediction accuracy Logistic Regression Model with Resampled training Data Machine Learning model for f1-score was 1% better than the Logistic Regression Model with the Original Data.


On the bases this analysis the The Logistic Regression Model with Resampled training Data Machine Learning is the best model to choose to process this loan company's data with.



















