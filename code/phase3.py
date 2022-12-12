# general imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# classifier imports
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# performance imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from yellowbrick.classifier import ROCAUC

# prevent warnings import
import warnings
warnings.filterwarnings("ignore")

# performance functions
## sensitivity (recall) function
def sensitivity(conf):
    return conf[1][1] / (conf[1][1] * conf[1][0])

## specificity function
def specificity(conf):
    return conf[0][0] / (conf[0][0] + conf[0][1])

# custom functions
## function to bin each country to their continent
### the following function performs binning
### from country to their respective continent
def country_to_continent(country):
    if country in africa:
        return 'Africa'
    elif country in asia:
        return 'Asia'
    elif country in australia:
        return 'Australia'
    if country in europe:
        return 'Europe'
    elif country in north_america:
        return 'North America'
    elif country in south_america:
        return 'South America'
    elif country in unk:
        return 'Unknown'
    elif country in Others:
        return 'Others'

# import data
original_csv = 'https://raw.githubusercontent.com/mc811mc/hotel-booking-demand/main/dataset/hotel_bookings.csv'

data = pd.read_csv(original_csv)
df = data.copy()

#################
# data cleaning #
#################

# fill missing values in the children column with '0'
df.children.fillna(0,inplace=True)

# fill missing values in the country column with 'Unknown'
df.country.fillna('Unknown',inplace=True)

# replace agent ids (numerical) values in the agent column with 'Agent' (categorical)
df.loc[df.agent.isnull() == False,'agent'] = 'Agent'

# fill null values in the agent column with 'No agent'
df.agent.fillna('No agent',inplace = True)

# put in 'Corporate' in the company column
# if market segment and distribution channel column is both 'Corporate'
df.loc[((df.market_segment == 'Corporate') | (df.distribution_channel == 'Corporate')) & (df.company.isnull()), 'company'] ='Corporate'

# replace non-null values in company column as 'Corporate'
df.loc[df.company.isnull() == False,'company'] = 'Corporate'

# fill in remaining and missing values in the company column with 'Individuals'
df.company.fillna('Individuals',inplace=True)

# correct the data type of the column arrival date year
df.arrival_date_year = df.arrival_date_year.astype(object)

# drop all duplicate rows in the dataset
df.drop_duplicates(inplace = True)

# this section handles binning countries into continents
# within the country of origin column.
# each country is categorized to one of the following: africa, asia, australia, europe, north america, south america,
# unknown, or others

## binning country column to its respective continents
africa = ['MOZ','BWA','MAR','ZAF','AGO','ZMB','ZWE','DZA','TUN','CAF','NGA','SEN','SYC','CMR','MUS','COM','UGA','CIV',
       'BDI','EGY','MWI','MDG','TGO','DJI','STP','ETH','RWA','BEN','TZA','GHA','KEN','GNB','BFA','LBY','MLI','NAM',
       'MRT','SDN','SLE']

asia = ['OMN','CN','IND','CHN','ISR','KOR','ARE','HKG','IRN','CYP','KWT','MDV','KAZ','PAK','IDN','LBN','PHL','AZE','BHR',
     'THA','MYS','ARM','JPN','LKA','JOR','SYR','SGP','SAU','VNM','QAT','UZB','NPL','MAC','TWN','IRQ','KHM','BGD','TJK',
     'TMP','MMR','LAO']

australia = ['AUS']

europe = ['PRT','GBR','ESP','IRL','FRA','ROU','NOR','POL','DEU','BEL','CHE','GRC','ITA','NLD','DNK','RUS','SWE','EST',
       'CZE','FIN','LUX','SVN','ALB','UKR','SMR','LVA','SRB','AUT','BLR','LTU','TUR','HUN','HRV','GEO','AND','SVK',
       'MKD','BIH','BGR','MLT','ISL','MCO','LIE','MNE']

north_america = ['USA', 'MEX', 'PRI', 'CRI', 'CUB', 'HND', 'NIC', 'GAB', 'PAN', 'SLV', 'GTM']

south_america = ['ARG', 'BRA', 'CHL', 'URY', 'COL', 'VEN', 'SUR', 'PER', 'ECU', 'BOL', 'PRY', 'GUY']

unk = ['Unknown']

Others = ['CYM','CPV','JAM','GIB','JEY','GGY','FJI','NZL','DOM','PLW','BHS','KNA','IMN','VGB','GLP','UMI','MYT','FRO',
       'BRB','ABW','AIA','DMA','PYF','LCA','ATA','ASM','NCL','KIR','ATF']

# modify country column to make only continents appear
df.country = df.country.apply(country_to_continent)

# dimensionality reduction
## create total_members column that includes children, babies, and adults
df['total_members'] = df.children + df.babies + df.adults

## create total nights column that includes stays in week nights and stays in weekend nights columns
df['total_nights'] = df.stays_in_week_nights + df.stays_in_weekend_nights

## drop the aggregated columns in the prior steps
df.drop(columns=['children', 'babies', 'adults', 'stays_in_week_nights', 'stays_in_weekend_nights'], inplace=True)

## drop irrelevant columns to improve model
df.drop(columns=['days_in_waiting_list', 'reservation_status','reservation_status_date'],inplace=True)

# feature encoding
df['hotel'] = df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})

df['meal'] = df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})

df['reserved_room_type'] = df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                   'L': 7, 'P': 8, 'B': 9})
df['deposit_type'] = df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})

# features to be dropped through regression analysis
df.drop(columns=['country', 'arrival_date_year', 'market_segment', 'agent', 'company'],inplace=True)
df.drop(df.loc[df['assigned_room_type']=='L'].index, inplace=True)
df.drop(df.loc[df['assigned_room_type']=='P'].index, inplace=True)
df.drop(df.loc[df['distribution_channel']=='Undefined'].index, inplace=True)

# feature selection
## get rid of rows that have no guests
df_temp=df.loc[~((df.total_members==0) & (df.is_canceled==0))]

## get rid of rows that have no number of nights stayed
df_temp=df_temp.loc[~((df_temp.total_nights==0) & (df_temp.is_canceled==0))]

df_preprocessed = pd.get_dummies(df_temp, drop_first=True)

# attribute variables
X = df_preprocessed.drop(columns='is_canceled')

# target variable
y = df_preprocessed.is_canceled

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#################
# decision tree #
#################
# decision tree initial case
def decision_tree_initial(X_train, X_test, y_train, y_test):
    tree_cls_no = DecisionTreeClassifier(random_state=0)
    tree_model_no = tree_cls_no.fit(X_train, y_train)

    y_pred_tree_no = tree_model_no.predict(X_test)

    tree_acc_no = accuracy_score(y_test, y_pred_tree_no)
    tree_conf_no = confusion_matrix(y_test, y_pred_tree_no)
    tree_clf_report_no = classification_report(y_test, y_pred_tree_no)

    # prints out the confusion matrix of the decision tree classifier
    print(f'Decision Tree (Initial) Confusion Matrix: \n{tree_conf_no}')
    disp = ConfusionMatrixDisplay(confusion_matrix=tree_conf_no,
                                  display_labels=tree_cls_no.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the decision tree classifier
    print(f'Decision Tree (Initial) Accuracy Score: {tree_acc_no:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the decision tree classifier
    print(f'Decision Tree (Initial) Classification Report: \n{tree_clf_report_no}')
    # prints out the specificity of the decision tree classifier
    print(f'Decision Tree (Initial) Specificity: {specificity(tree_conf_no):.3f}')
    # prints out the ROC curve for the decision tree classifier
    ran_for_visualizer = ROCAUC(tree_cls_no)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()

#######################
# logistic regression #
#######################
def logistic_regression(X_train, X_test, y_train, y_test):
    log_reg_cls = LogisticRegression(random_state=0)
    log_reg_model = log_reg_cls.fit(X_train, y_train)

    y_pred_log_reg = log_reg_model.predict(X_test)

    log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
    log_reg_conf = confusion_matrix(y_test, y_pred_log_reg)
    log_reg_clf_report = classification_report(y_test, y_pred_log_reg)

    # prints out the confusion matrix of the logistic regression classifier
    print(f'Logistic Regression Confusion Matrix: \n{log_reg_conf}')
    disp = ConfusionMatrixDisplay(confusion_matrix=log_reg_conf,
                                  display_labels=log_reg_cls.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the logistic regression classifier
    print(f'Logistic Regression Accuracy Score: {log_reg_acc:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the logistic regression classifier
    print(f'Logistic Regression Classification Report: \n{log_reg_clf_report}')
    # prints out the specificity of the logistic regression classifier
    print(f'Logistic Regression Specificity: {specificity(log_reg_conf):.3f}')
    # prints out the ROC curve for the logistic regression classifier
    ran_for_visualizer = ROCAUC(log_reg_cls)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()

#######
# knn #
#######
# knn initial case
def knn_initial(X_train, X_test, y_train, y_test):
    knn_cls = KNeighborsClassifier()
    knn_model = knn_cls.fit(X_train, y_train)

    y_pred_knn = knn_model.predict(X_test)

    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_conf = confusion_matrix(y_test, y_pred_knn)
    knn_clf_report = classification_report(y_test, y_pred_knn)

    # prints out the confusion matrix of the knn classifier
    print(f'KNN (Default) Confusion Matrix: \n{knn_conf}')
    disp = ConfusionMatrixDisplay(confusion_matrix=knn_conf,
                                  display_labels=knn_cls.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the knn classifier
    print(f'KNN (Default) Accuracy Score: {knn_acc:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score of the knn classifier
    print(f'KNN (Default) Classification Report: \n{knn_clf_report}')
    # prints out the specificity of the knn classifier
    print(f'KNN (Default) Specificity: {specificity(knn_conf):.3f}')
    # prints out the ROC curve for the knn classifier
    ran_for_visualizer = ROCAUC(knn_cls)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()

#######
# svm #
#######
def svm(X_train, X_test, y_train, y_test):
    svm_cls = SVC(random_state=0)
    svm_model = svm_cls.fit(X_train, y_train)

    y_pred_svm = svm_model.predict(X_test)

    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_conf = confusion_matrix(y_test, y_pred_svm)
    svm_clf_report = classification_report(y_test, y_pred_svm)

    # prints out the confusion matrix of the svm classifier
    print(f'SVM Confusion Matrix: \n{svm_conf}')
    disp = ConfusionMatrixDisplay(confusion_matrix=svm_conf,
                                  display_labels=svm_cls.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the svm classifier
    print(f'SVM Accuracy Score: {svm_acc:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the svm classifier
    print(f'SVM Classification Report: \n{svm_clf_report}')
    # prints out the specificity of the svm classifier
    print(f'SVM Specificity: {specificity(svm_conf):.3f}')

###############
# naive bayes #
###############
def naive_bayes(X_train, X_test, y_train, y_test):
    gaussian_cls = GaussianNB()
    gaussian_model = gaussian_cls.fit(X_train, y_train)

    y_pred_gaussian = gaussian_model.predict(X_test)

    gaussian_acc = accuracy_score(y_test, y_pred_gaussian)
    gaussian_conf = confusion_matrix(y_test, y_pred_gaussian)
    gaussian_clf_report = classification_report(y_test, y_pred_gaussian)

    # prints out the confusion matrix of the naive bayes classifier
    print(f'Naive Bayes Confusion Matrix: \n{gaussian_conf}')
    disp = ConfusionMatrixDisplay(confusion_matrix=gaussian_conf,
                                  display_labels=gaussian_cls.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the naive bayes classifier
    print(f'Naive Bayes Accuracy Score: {gaussian_acc:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the naive bayes classifier
    print(f'Naive Bayes Classification Report: \n{gaussian_clf_report}')
    # prints out the specificity of the naive bayes classifier
    print(f'Naive Bayes Specificity: {specificity(gaussian_conf):.3f}')
    # prints out the ROC curve for the naive bayes classifier
    ran_for_visualizer = ROCAUC(gaussian_cls)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()


#################
# random forest #
#################
# random forest using gini (default)
def random_forest_default(X_train, X_test, y_train, y_test):
    ran_for_cls_gini = RandomForestClassifier(random_state=0, criterion='gini')
    ran_for_model_gini = ran_for_cls_gini.fit(X_train, y_train)

    y_pred_ran_for_gini = ran_for_model_gini.predict(X_test)

    ran_for_acc_gini = accuracy_score(y_test, y_pred_ran_for_gini)
    ran_for_conf_gini = confusion_matrix(y_test, y_pred_ran_for_gini)
    ran_for_clf_report_gini = classification_report(y_test, y_pred_ran_for_gini)

    # prints out the confusion matrix of the random forest classifier
    print(f'Random Forest Confusion Matrix: \n{ran_for_conf_gini}')
    disp = ConfusionMatrixDisplay(confusion_matrix=ran_for_conf_gini,
                                  display_labels=ran_for_cls_gini.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the random forest classifier
    print(f'Random Forest Accuracy Score: {ran_for_acc_gini:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the random forest classifier
    print(f'Random Forest Classification Report: \n{ran_for_clf_report_gini}')
    # prints out the specificity of the random forest classifier
    print(f'Random Forest Specificity: {specificity(ran_for_conf_gini):.3f}')
    # prints out the ROC curve for the random forest classifier
    ran_for_visualizer = ROCAUC(ran_for_cls_gini)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()

# random forest using entropy
def random_forest_entropy(X_train, X_test, y_train, y_test):
    ran_for_cls_entropy = RandomForestClassifier(random_state=0, criterion='entropy')
    ran_for_model_entropy = ran_for_cls_entropy.fit(X_train, y_train)

    y_pred_ran_for_entropy = ran_for_model_entropy.predict(X_test)

    ran_for_acc_entropy = accuracy_score(y_test, y_pred_ran_for_entropy)
    ran_for_conf_entropy = confusion_matrix(y_test, y_pred_ran_for_entropy)
    ran_for_clf_report_entropy = classification_report(y_test, y_pred_ran_for_entropy)

    # prints out the confusion matrix of the random forest classifier
    print(f'Random Forest (Entropy) Confusion Matrix: \n{ran_for_conf_entropy}')
    disp = ConfusionMatrixDisplay(confusion_matrix=ran_for_conf_entropy,
                                  display_labels=ran_for_cls_entropy.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the random forest classifier
    print(f'Random Forest (Entropy) Accuracy Score: {ran_for_acc_entropy:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the random forest classifier
    print(f'Random Forest (Entropy) Classification Report: \n{ran_for_clf_report_entropy}')
    # prints out the specificity of the random forest classifier
    print(f'Random Forest (Entropy) Specificity: {specificity(ran_for_conf_entropy):.3f}')
    # prints out the ROC curve for the random forest classifier
    ran_for_visualizer = ROCAUC(ran_for_cls_entropy)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()

# random forest using log loss
def random_forest_log_loss(X_train, X_test, y_train, y_test):
    ran_for_cls_log = RandomForestClassifier(random_state=0, criterion='log_loss')
    ran_for_model_log = ran_for_cls_log.fit(X_train, y_train)

    y_pred_ran_for_log = ran_for_model_log.predict(X_test)

    ran_for_acc_log = accuracy_score(y_test, y_pred_ran_for_log)
    ran_for_conf_log = confusion_matrix(y_test, y_pred_ran_for_log)
    ran_for_clf_report_log = classification_report(y_test, y_pred_ran_for_log)

    # prints out the confusion matrix of the random forest classifier
    print(f'Random Forest (Log Loss) Confusion Matrix: \n{ran_for_conf_log}')
    disp = ConfusionMatrixDisplay(confusion_matrix=ran_for_conf_log,
                                  display_labels=ran_for_cls_log.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the random forest classifier
    print(f'Random Forest (Log Loss) Accuracy Score: {ran_for_acc_log:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the random forest classifier
    print(f'Random Forest (Log Loss) Classification Report: \n{ran_for_clf_report_log}')
    # prints out the specificity of the random forest classifier
    print(f'Random Forest (Log Loss) Specificity: {specificity(ran_for_conf_log):.3f}')
    # prints out the ROC curve for the random forest classifier
    ran_for_visualizer = ROCAUC(ran_for_cls_log)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()

##################
# neural network #
##################
def neural_network(X_train, X_test, y_train, y_test):
    neural_cls = MLPClassifier(random_state=0)
    neural_model = neural_cls.fit(X_train, y_train)

    y_pred_neural = neural_model.predict(X_test)

    neural_acc = accuracy_score(y_test, y_pred_neural)
    neural_conf = confusion_matrix(y_test, y_pred_neural)
    neural_clf_report = classification_report(y_test, y_pred_neural)

    # prints out the confusion matrix of the neural network classifier
    print(f'Neural Network  Confusion Matrix: \n{neural_conf}')
    disp = ConfusionMatrixDisplay(confusion_matrix=neural_conf,
                                  display_labels=neural_cls.classes_)
    disp.plot()
    plt.show()
    # prints out the accuracy score of the neural network classifier
    print(f'Neural Network  Accuracy Score: {neural_acc:.3f}')
    # prints out the precision, recall (sensitivity), and f1 score
    # of the neural network classifier
    print(f'Neural Network Classification Report: \n{neural_clf_report}')
    # prints out the specificity of the neural network classifier
    print(f'Neural Network Specificity: {specificity(neural_conf):.3f}')
    # prints out the ROC curve for the neural network classifier
    ran_for_visualizer = ROCAUC(neural_cls)
    # fit the training data to the visualizer
    ran_for_visualizer.fit(X_train, y_train)
    # evaluate the model on the test data
    ran_for_visualizer.score(X_test, y_test)
    # display the visualizer
    ran_for_visualizer.show()

# run individual classifiers
decision_tree_initial(X_train, X_test, y_train, y_test)
logistic_regression(X_train, X_test, y_train, y_test)
knn_initial(X_train, X_test, y_train, y_test)
svm(X_train, X_test, y_train, y_test)
naive_bayes(X_train, X_test, y_train, y_test)
random_forest_default(X_train, X_test, y_train, y_test)
random_forest_entropy(X_train, X_test, y_train, y_test)
random_forest_log_loss(X_train, X_test, y_train, y_test)
neural_network(X_train, X_test, y_train, y_test)