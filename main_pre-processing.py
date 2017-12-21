#Title: Baseball Pitch Prediction Model
#Author: Matthew Spitz
#Date: 12/21/17

from functions import *

#change paths corresponding to your local machine if nec.
input_file_1 = "2016_team_batting_data.csv"
input_file_2a = "2016_player_batting_data.csv"
input_file_2b = "2016_player_pitching_data.csv"
input_file_3a = "2016_player_batting_data_clean.csv"
input_file_3b = "2016_player_pitching_data_clean.csv"
input_file_4 = "2016_season_data_combined.csv"
input_file_5 = "2016_team_batting_data_clean.csv"
input_file_6 = "2016_player_batting_data_clean2.csv"
input_file_7 = "2016_player_pitching_data_clean2.csv"

answer = raw_input("Would you like to extract the features of the model? Answer Yes or No: ")

if answer.lower() == 'no':
    
    answer_2 = raw_input("Would you like to predict pitch type for a specific event? Answer Yes or No: ")
    
    if answer_2.lower() == 'no':
        print "Thank you for participating in the pitch prediction project"
    elif answer_2.lower() == 'yes':
        d1,d2,d3 = get_dicts(input_file_4)
        get_player_data(d1,d2,d3)

elif answer.lower() == 'yes':
    clean_baseball_reference_team_batting(input_file_1)
    clean_name_baseball_reference_player(input_file_2a)
    clean_name_baseball_reference_player(input_file_2b)
    clean_multiples_baseball_reference_player(input_file_3a)
    clean_multiples_baseball_reference_player(input_file_3b)

    pitch_speed,pitch_zone = calculate_speed_zone_average(input_file_4)
    extract_features(input_file_4,input_file_5,input_file_6,input_file_7,pitch_speed,pitch_zone)

    answer_2 = raw_input("Would you like to predict pitch type of a specific event? Answer Yes or No: ")
        
    if answer_2.lower() == 'no':
        print "Thank you for participating in the pitch prediction project"
    elif answer_2.lower() == 'yes':
        d1,d2,d3 = get_dicts(input_file_4)
        get_player_data(d1,d2,d3)
