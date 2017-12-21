#Title: Baseball Pitch Prediction Model
#Author: Matthew Spitz
#Date: 12/21/17

from functions import *
import sys

###RUN IN PYSPARK###

#change paths corresponding to your local machine if nec.
personal_path = "/home/matthew/Dropbox/Documents/Columbia/2017-18_Fall/2-BigData/HW/Project/"

input_file_1 = "2016_season_data_combined_clean_binary.csv"
input_file_2 = "/vector_form/part-00000"

answer=sys.argv[1]
answer_2=sys.argv[2]

if answer.lower() == 'no':

    if answer_2.lower() == 'no':
        print "Thank you for participating in the pitch prediction project"
    
    elif answer_2.lower() == 'yes':
        df,spark = create_spark_df(personal_path+"individual_prediction.csv")
        create_vector_file(df,personal_path,"vector_pred/")
        model,test = train_model(personal_path+input_file_2,spark)
        predictions = make_predictions(model,personal_path+"/vector_pred/part-00000",personal_path,spark)

elif answer.lower() == 'yes':
    df,spark = create_spark_df(personal_path+input_file_1)
    create_vector_file(df,personal_path,"vector_form/")
    model,test = train_model(personal_path+input_file_2,spark)
    predictions = make_predictions(model,test,personal_path,spark)
    accuracy(predictions)

    if answer_2.lower() == 'no':
        print "Thank you for participating in the pitch prediction project"
    
    elif answer_2.lower() == 'yes':
        df,spark = create_spark_df(personal_path+"individual_prediction.csv")
        create_vector_file(df,personal_path,"vector_pred/")
        model,test = train_model(personal_path+input_file_2,spark)
        predictions = make_predictions(model,personal_path+"/vector_pred/part-00000",personal_path,spark)
