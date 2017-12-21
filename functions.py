#Title: Pitch Prediction
#Author: Matthew Spitz
#Date: 12/21/17

import csv
import difflib

'''
Create team names from Baseball Reference abbrevations
to corresponding names in Sportsradar dataset for matching
'''
def clean_baseball_reference_team_batting(input_file):
    input_f = open(input_file,'r')
    reader = csv.reader(input_f)
    output_f = open("2016_team_batting_data_clean.csv",'w')
    writer = csv.writer(output_f, delimiter=',')

    reader.next()
    for row in reader:
        data = row
        if row[0] == "ARI":
            data.append("Diamondbacks")
        elif row[0] == "COL":
            data.append("Rockies")
        elif row[0] == "TOR":
            data.append("Blue Jays")
        elif row[0] == "TBR":
            data.append("Rays")
        elif row[0] == "CHC":
            data.append("Cubs")
        elif row[0] == "SFG":
            data.append("Giants")
        elif row[0] == "BOS":
            data.append("Red Sox")
        elif row[0] == "SDP":
            data.append("Padres")
        elif row[0] == "TEX":
            data.append("Rangers")
        elif row[0] == "MIL":
            data.append("Brewers")
        elif row[0] == "LAD":
            data.append("Dodgers")
        elif row[0] == "WSN":
            data.append("Nationals")
        elif row[0] == "MIA":
            data.append("Marlins")
        elif row[0] == "NYM":
            data.append("Mets")
        elif row[0] == "PIT":
            data.append("Pirates")
        elif row[0] == "STL":
            data.append("Cardinals")
        elif row[0] == "ATL":
            data.append("Braves")
        elif row[0] == "KCR":
            data.append("Royals")
        elif row[0] == "CIN":
            data.append("Reds")
        elif row[0] == "HOU":
            data.append("Astros")
        elif row[0] == "LAA":
            data.append("Angels")
        elif row[0] == "OAK":
            data.append("Athletics")
        elif row[0] == "CHW":
            data.append("White Sox")
        elif row[0] == "CLE":
            data.append("Indians")
        elif row[0] == "MIN":
            data.append("Twins")
        elif row[0] == "PHI":
            data.append("Phillies")
        elif row[0] == "NYY":
            data.append("Yankees")
        elif row[0] == "SEA":
            data.append("Mariners")
        elif row[0] == "BAL":
            data.append("Orioles")
        elif row[0] == "DET":
            data.append("Tigers")
    
        writer.writerow(data)

'''
Clean player name in Baseball Reference dataset
'''
def clean_name_baseball_reference_player(input_file):
    input_f = open(input_file,'r')
    reader = csv.reader(input_f)
    if input_file.split("_")[2] == "batting":
        output_f = open("2016_player_batting_data_clean.csv",'w')
    else:
        output_f = open("2016_player_pitching_data_clean.csv",'w')
    writer = csv.writer(output_f, delimiter=',')

    reader.next()
    for row in reader:
        data = row
        nm = row[1].split(" ")
        cnt = 0
        for letter in nm[1]:
            if letter.isalpha():
                cnt += 1
            else:
                break
        last_nm = nm[1][:cnt]
        full_nm = nm[0] + " " + last_nm
        data.append(full_nm)
        writer.writerow(data)

'''
Remove duplicate names in Baseball Reference dataset
'''
def clean_multiples_baseball_reference_player(input_file):
    input_f = open(input_file,'r')
    reader = csv.reader(input_f)
    if input_file.split("_")[2] == "batting":
        output_f = open("2016_player_batting_data_clean2.csv",'w')
    else:
        output_f = open("2016_player_pitching_data_clean2.csv",'w')
    writer = csv.writer(output_f, delimiter=',')

    name = ""
    last_nm = ""
    doubles = []

    for row in reader:
        name = row[len(row)-1]

        if name == last_nm and name not in doubles and last_rw[2] == row[2]:
            doubles.append(name)
        elif name not in doubles:
            writer.writerow(row)

        last_nm = row[len(row)-1]
        last_rw = row

'''
Calculate 2016 average pitch speed and pith location for
each pitcher in Sportsradar dataset
'''
def calcuate_speed_zone_average(input_file):
    input_f = open(input_file,'r')
    reader = csv.reader(input_f)

    speed_avg = {}
    zone_avg = {}
    for row in reader:
        if row[36] == "PITCH" and row[53] != "IB" and row[53] != "PI" and row[53] != "KN" and row[54] != "Other":

            nm = row[50] + " " + row[51]
            speed = float(row[55])
            zone = float(row[56])
            if nm not in speed_avg and nm:
                speed_avg[nm] = [speed]
            elif nm:
                speed_avg[nm].append(speed)
            if nm not in zone_avg and nm:
                zone_avg[nm] = [zone]
            elif nm:
                zone_avg[nm].append(zone)

    avg_speed = {}
    avg_zone = {}

    for k,v in speed_avg.items():
        total = 0
        for speed in v:
            total += speed
        avg_speed[k] = total / len(v)

    for k,v in zone_avg.items():
        total = 0
        for zone in v:
            total += zone
        avg_zone[k] = total / len(v)

    return avg_speed, avg_zone

'''
Extract feature set
'''
def extract_features(input_file_1, input_file_2, input_file_3, input_file_4, avg_speed, avg_zone):
    input_f = open(input_file_1,'r')
    reader = csv.reader(input_f)
    output_f = open("2016_season_data_combined_clean_binary.csv",'w')
    writer = csv.writer(output_f, delimiter=',')

    input_f_2 = open(input_file_2,'r')
    reader_2 = csv.reader(input_f_2)

    input_f_3 = open(input_file_3,'r')
    reader_3 = csv.reader(input_f_3)

    input_f_4 = open(input_file_4,'r')
    reader_4 = csv.reader(input_f_4)

    team_OPS = {}
    for row in reader_2:
        team_OPS[row[len(row)-1]] = row[20]

    player_OPS = {}
    for row in reader_3:
        if row[21]:
            player_OPS[row[len(row)-1]] = row[21]
    player_OPS_list = [keys for keys in player_OPS.keys()]
    
    player_BIP = {}
    for row in reader_4:
        player_BIP[row[len(row)-1]] = row[13][:len(row[13])-1]
    player_BIP_list = [keys for keys in player_BIP.keys()]


    pitch = {}
    venue = {}
    hitters = {}
    pitchers = {}

    cnt = 0
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0

    total = 0
    for row in reader:
    
        if row[36] == "PITCH" and row[53] != "IB" and row[53] != "PI" and row[53] != "KN" and row[54] != "Other":
            clean_row = []
        
            #define features
            pitch_id = row[53]
            venue_name = row[15]
            inning = int(row[31])
            inning_event_count = int(row[34])
            hitter_id = row[43]
            hitter_wgt = int(row[46])
            hitter_hgt = int(row[47])
            hitter_side = row[48]
            pitcher_id = row[49]
            pitcher_hand = row[52]
            pitcher_count = int(row[57])
            hit_pitch_count = int(row[58])
            balls_count = int(row[61])
            strikes_count = int(row[62])
            outs_count = int(row[63])
            home_score = int(row[97])
            away_score = int(row[98])
            home_team_nm = row[13]
            away_team_nm = row[11]
            hitter_nm = row[45] + " " + row[44]
            hitter_nm = difflib.get_close_matches(hitter_nm, player_OPS_list, n=1)
            pitcher_nm = row[50] + " " + row[51]
            pitch_speed = avg_speed[pitcher_nm]
            pitch_zone = avg_zone[pitcher_nm]
            pitcher_nm = difflib.get_close_matches(pitcher_nm, player_BIP_list, n=1)

            if row[32] == "TOP":
                run_diff = home_score - away_score
                try:
                    batting_tm_OPS = float(team_OPS[away_team_nm])
                except:
                    batting_tm_OPS = 0.739
                    print 'away team OPS not found'
                    print away_team_nm
                
            elif row[32] == "BOT":
                run_diff = away_score - home_score
                try:
                    batting_tm_OPS = float(team_OPS[home_team_nm])
                except:
                    batting_tm_OPS = 0.739
                    print 'home team OPS not found'
                    print home_team_nm

            try:
                hitter_OPS = float(player_OPS[hitter_nm[0]])
            except:
                print 'no hitter OPS found'
                print row[45] + " " + row[44]
                hitter_OPS = 0.735
            try:
                pitcher_BIP = float(player_BIP[pitcher_nm[0]])
            except:
                print 'no pitcher BIP found'
                print row[50] + " " + row[51]
                pitchter_BIP = 28.3
        
            #assign variables for categorical data
            if pitch_id not in pitch and pitch_id:
                pitch[pitch_id] = cnt
                cnt += 1
            if venue_name not in venue and venue_name:
                venue[venue_name] = cnt_1
                cnt_1 += 1
            if hitter_id not in hitters and hitter_id:
                hitters[hitter_id] = cnt_2
                cnt_2 += 1
            if pitcher_id not in pitchers and pitcher_id:
                pitchers[pitcher_id] = cnt_3
                cnt_3 += 1
            if hitter_side == "L":
                hitter_side = 0
            elif hitter_side == "R":
                hitter_side = 1
            elif hitter_side == "B":
                hitter_side = 2
            if pitcher_hand== "L":
                pitcher_hand = 0
            elif pitcher_hand == "R":
                pitcher_hand = 1
            elif pitcher_hand == "B":
                pitcher_hand = 2
        
            #add new data to new row for output file
            try:
                clean_row.append(pitchers[pitcher_id])
            except:
                print "no pitcher", row
                clean_row.append("")
            
            clean_row.append(pitcher_hand)
            clean_row.append(pitcher_count)
        
            try:
                clean_row.append(hitters[hitter_id])
            except:
                print "no hitter",row
                clean_row.append("")
            
            clean_row.append(hitter_side)
            clean_row.append(hit_pitch_count)
            clean_row.append(hitter_wgt)
            clean_row.append(hitter_hgt)
            clean_row.append(inning)
            clean_row.append(inning_event_count)
            clean_row.append(balls_count)
            clean_row.append(strikes_count)
            clean_row.append(outs_count)
            clean_row.append(run_diff)
            clean_row.append(batting_tm_OPS)
            clean_row.append(hitter_OPS)
            clean_row.append(pitcher_BIP)
            clean_row.append(pitch_speed)
            clean_row.append(pitch_zone)
        
            try:
                clean_row.append(venue[venue_name])
            except:
                print "no venue",row
                clean_row.append("") 
            
            try:
                if pitch_id == "FA":
                    clean_row.append(0)
                else:
                    clean_row.append(1)
            except:
                print "no pitch"
                clean_row.append("")
        
            writer.writerow(clean_row)

'''
Get player and venue name dictionaries
for individual pitch predictions requested by user
'''
def get_dicts(input_file):
    input_f = open(input_file,'r')
    reader = csv.reader(input_f)
   
    pitchers = {}
    hitters = {}
    venue = {}
    
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    for row in reader:
        if row[36] == "PITCH" and row[53] != "IB" and row[53] != "PI" and row[53] != "KN" and row[54] != "Other":
            hitter_nm = row[45] + " " + row[44]
            pitcher_nm = row[50] + " " + row[51]
            venue_nm = row[15]
            
            if venue_nm not in venue and venue_nm:
                venue[venue_nm] = cnt_1
                cnt_1 += 1
            if hitter_nm not in hitters and hitter_nm:
                hitters[hitter_nm] = cnt_2
                cnt_2 += 1
            if pitcher_nm not in pitchers and pitcher_nm:
                pitchers[pitcher_nm] = cnt_3
                cnt_3 += 1

    return pitchers,hitters,venue

'''
Create testing data for individual pitch predictions
requested by the user
'''
def get_player_data(pitchers,hitters,venue):
    
    output_f = open("individual_prediction.csv",'w')
    writer = csv.writer(output_f, delimiter=',')

    while True:
        pitcher = raw_input("What pitcher would you like to predict pitch type for? (Enter Full Name): ")
        try:
            pitcher = pitchers[pitcher]
            break
        except:
            print "pitcher name not found - try again"

    pitcher_hand = raw_input("What hand does the pitcher throw with? (Enter Right, Left, or Both): ")
    if pitcher_hand.lower() == "left":
        pitcher_hand = 0
    elif pitcher_hand.lower() == "right":
        pitcher_hand = 1
    else:
        pitcher_hand = 2
    
    pitcher_count = raw_input(
            "How many pitches has the pitcher thrown thus far in the game? (Enter integer number): ")
    
    while True:
        hitter = raw_input("Who is the batter that the pitcher is facing? (Enter Full Name): ")
        try:
            hitter = hitters[hitter]
            break
        except:
            print "hitter name not found - try again"
    
    hitter_side = raw_input("What side of the plate does the batter hit from? (Enter Right, Left, or Both): ")
    if hitter_side.lower() == "left":
        hitter_side = 0
    elif hitter_side.lower() == "right":
        hitter_side = 1
    else:
        hitter_side = 2

    hit_pitch_count = raw_input(
            "How many pitches has the batter faced thus far in the at bat? (Enter integer number): ")
    hitter_wgt = raw_input("What is the weight of the batter? (Enter integer number): ")
    hitter_hgt = raw_input("What is the height of the batter? (Enter integer number): ")
    inning = raw_input("What inning is the game in? (Enter integer number): ")
    inning_event_count = raw_input(
            "How many batters has the pitcher faced in the inning, including the current batter? (Enter integer number): ")
    balls_count = raw_input(
            "How many balls has the pitcher thrown thus far in the at bat? (Enter integer number): ")
    strikes_count = raw_input(
            "How many strikes has the pitch thrown thus far in the at bat? (Enter integer number): ")
    outs_count = raw_input("How many outs are there in the inning currently? (Enter integer number): ")
    run_diff = raw_input(
            "How many runs is the pitching team winning or losing by? (Enter 0 for tie and negative integer if losing and postive integer if winning): ")
    batting_tm_OPS = raw_input("What is the average OPS of the batting team? (Enter float number): ")
    hitter_OPS = raw_input("What is the average OPS of the batter? (Enter float number): ")
    pitcher_BIP = raw_input("What is the average Balls in Play % of the pitcher? (Enter float number): ")
    pitch_speed = raw_input("What is the average speed of the pitcher's pitches? (Enter float number): ")
    pitch_zone = raw_input("What is the average location of the pitcher's pitches? (Enter float number): ")
    
    while True:
        venue_n = raw_input("What stadium is the game being played in? (Enter full venue name): ")
        try:
            venue_n = venue[venue_n]
            break
        except:
            print "venue name not found - try again"
    
    writer.writerow([pitcher, pitcher_hand, pitcher_count, hitter, hitter_side, hit_pitch_count, hitter_wgt, 
        hitter_hgt, inning, inning_event_count, balls_count, strikes_count, outs_count, run_diff, batting_tm_OPS, 
        hitter_OPS, pitcher_BIP, pitch_speed, pitch_zone, venue_n, 100])


#####SPARK#####
'''
Create Spark DataFrame of features and label
'''
def create_spark_df(input_file):
    from pyspark.conf import SparkConf
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.config(conf=SparkConf()).appName("BigData").getOrCreate()

    from pyspark.sql import Row

    rows = spark.read.text(input_file).rdd
    split_lines = rows.map(lambda row: row.value.split(","))

    pitchRDD = split_lines.map(lambda x:Row(pitcher=int(x[0]),pitch_hand=int(x[1]),pitch_count=int(x[2]),\
            hitter=int(x[3]),hitter_side=int(x[4]),hitter_count=int(x[5]),hitter_wgt=int(x[6]),\
            hitter_hgt=int(x[7]),inning=int(x[8]),inning_batter_count=int(x[9]),balls_count=int(x[10]),\
            strikes_count=int(x[11]),outs_count=int(x[12]),run_diff=int(x[13]),batting_tm_OPS=float(x[14]),\
            hitter_OPS=float(x[15]),pitcher_BIP=float(x[16]),pitch_speed=float(x[17]),pitch_zone=float(x[18]),\
            venue=int(x[19]),pitch_id=int(x[20])))

    pitch_outcome = spark.createDataFrame(pitchRDD)
    pitch_outcome.show()
    
    return pitch_outcome,spark

'''
Create vector file needed to for Spark MLlib
'''
def create_vector_file(pitch_outcome,path,folder):
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.util import MLUtils

    pitch_o_RDD = pitch_outcome.rdd
    x = pitch_o_RDD.map(lambda data: LabeledPoint(data[13],[data[0],data[1],data[2],data[3],\
            data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[14],\
            data[15],data[16],data[17],data[18],data[19],data[20]]))
    MLUtils.saveAsLibSVMFile(x, path+folder)

'''
Train machine learning model
'''
def train_model(input_file,spark):
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml import Pipeline

    data = spark.read.format("libsvm").load(input_file)
    (training,test) = data.randomSplit([0.8,0.2])

    gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
    pipeline = Pipeline(stages=[gbt])

    model = pipeline.fit(training)
    
    print model.stages[0].featureImportances

    return model,test

'''
Make predictions on testing data
'''
def make_predictions(model,test,path,spark):
    if type(test) == str:
        test_data = spark.read.format("libsvm").load(test)
    else:
        test_data = test

    predictions = model.transform(test_data)
    predictions.select("prediction", "label", "features").show(5)
    predictions.toPandas().to_csv(path+'pitch_predictions.csv')
    
    return predictions

'''
Calculate accuracy of model
'''
def accuracy(predictions):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    print "Accuracy = %g" % (accuracy)

##########

'''
Create file with testing data plus predicted pitch
Used in Tableau dashboard
'''
def create_predictions_file(input_file):
    input_f = open(input_file,'r')
    reader = csv.reader(input_f)
    output_f = open("pitch_predictions_clean.csv",'w')
    writer = csv.writer(output_f, delimiter=',')

    columns = ['balls_count','batting_tm_OPS','hitter','hitter_OPS','hitter_count','hitter_hgt','hitter_side',\
            'hitter_wgt','inning','inning_batter_count','outs_count','pitch_count','pitch_hand','pitch_speed',\
            'pitch_zone','pitcher','pitcher_BIP','run_diff','strikes_count','venue']
    writer.writerow(columns)
    reader.next()

    index_cnt = 0
    for row in reader:
        dict_temp = {}
        final_data = []

        new_row = row[1].replace(']','[').split('[')
        index = new_row[1].split(',')
        data = new_row[3].split(',')

        cnt = 0
        for item in index:
            dict_temp[columns[int(item)]] = data[cnt]
            cnt += 1

        for item in columns:
            try:
                final_data.append(dict_temp[item])
            except:
                final_data.append(0.0)

        final_data.append(row[len(row)-1])

        writer.writerow(final_data)

        index_cnt += 1
