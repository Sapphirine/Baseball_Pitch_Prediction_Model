# Baseball Pitch Prediction Model


## Files in Repo:
   
   1. functions.py
   2. main_pre-processing.py
   3. main_model.py

## Datsets Required (before running code):
  
   1. Sportradar 2016 MLB Data via Google BigQuery
        - source: https://cloud.google.com/bigquery/public-data/baseball
        1. export "games_wide" and "games_post_wide" tables to preferred storage application (e.g. Google Cloud Platform Storage)
        2. save data as 2 CSV files on local machine
        3. combine CSV files using 'merge' command in terminal 

   2. Baseball Reference - 3 tables
        - source: https://www.baseball-reference.com/leagues/MLB/2016-standard-batting.shtml
        1. use the 'Get table as CSV' function to save the "Team Standard Batting" and "Player Standard Batting" tables to your local machine
        <br> 
        - source: https://www.baseball-reference.com/leagues/MLB/2016-pitches-pitching.shtml
        1. use the 'Get tableas CSV' function to save the "Player Pitching Pitches" table to your local machine

## Languages / Tools / Frameworks / Software Installation Required:
   
   1. Python 2.. version
        - packages: csv, sys, difflib
   2. Spark 2... version
   3. Ubuntu 

## How to use files: https://www.youtube.com/watch?v=egHaDoKrg8c

