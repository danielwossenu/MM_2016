import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import csv
import GetTourney as GT
data = {}
filereader = csv.reader(open("RegularSeasonDetailedResults.csv"), delimiter=",")
header = filereader.next()

#This for loop goes through each game and creates an item in "data" for each team that has schedule=[] and season=[]
#schedule is append with [opponent, win/loss, team score, opponent's score] for the winning and losing team for each matchup
#season is just [] for now, will be filled with stats later

def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)

    """
    This is originally from from:
    http://zurb.com/forrst/posts/An_Elo_Rating_function_in_Python_written_for_foo-hQl
    """
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + 10**exp)
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank

all_games = []
for game in filereader:
    all_games.append(game)


#attaching ELO scores to the game results
elo_dict={}
for game in all_games:
    year = game[0]
    if year not in elo_dict:
        elo_dict[year] = {}
    wteam = game[2]
    lteam = game[4]

    if wteam not in elo_dict[year]:
        elo_dict[year][wteam] = 1600
    if lteam not in elo_dict[year]:
        elo_dict[year][lteam] = 1600

    winner_rank = elo_dict[year][wteam]
    loser_rank = elo_dict[year][lteam]

    game.append(winner_rank)
    game.append(loser_rank)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + 10 ** exp)
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    elo_dict[year][wteam] = new_winner_rank
    elo_dict[year][lteam] = new_loser_rank



for game in all_games:
    if game[0] not in data:
        data[game[0]]={}
    year = game[0]
    wteam = game[2]
    lteam= game[4]
    if wteam not in data[year]:
        data[year][wteam] = {'schedule':[], 'season':[]}

    a = [lteam,1, game[3], game[5],game[-2],game[-1]]
    a = [int(x) for x in a]
    # a.extend(game[8:21])
    data[year][wteam]['schedule'].append(a)

    if lteam not in data[year]:
        data[year][lteam] = {'schedule': [], 'season':[]}
    b = [wteam,0, game[5], game[3],game[-1],game[-2]]
    b = [int(x) for x in b]
    # b.extend(game[21:14])
    # data[year][wteam]['schedule'].append([wteam,0, game[5]])
    data[year][lteam]['schedule'].append(b)



# year = '2003'
years = data.keys()


#these next 3 nested for loops:
# 1st(outer loop) goes through each year
# 2nd(1st inner loop)for each team, calcs their average pts/game and pts against/game and saves in "data" for each team under "season"
# 3rd(2nd inner loop) goes through teams again and calcs offensive power and defensive suppression for each team and saves in "data" for each team under "season"
#Offensive power(OP): if you're oppenents collectively on average give up 100pts/ game and you're averaging 103, you have 1.3 offensive power
#Defensive Suppresion(DS): if you're on average give up 75pts/ game and your opponents collectively on average score 100pts/game, you have .75 defensive suppression
#OP: higher is better. if >1 then you are scoring more than you're oppenents usually allow
#DS: lower is better. if >1 then you are allowing your oppenents to score more on you than they usually do
for year in years:
    teams = data[year].keys()
    for team in teams:
        sche = data[year][team]['schedule']
        print len(sche)
        offtot=0
        deftot=0
        for game in sche:
            offtot += game[2]
            deftot += game[3]
        print team
        data[year][team]['season'].append(offtot/float(len(sche)))
        data[year][team]['season'].append(deftot / float(len(sche)))
        # print sche
        # print [sum(x) for x in zip(*data[year][team]['schedule'])]


    for team in teams:
        sche = data[year][team]['schedule']
        print len(sche)
        oppofftot=0
        oppdeftot=0
        reg_season_elo = 0
        for game in sche:
            opp = game[0]
            oppofftot += data[year][str(opp)]['season'][0]
            oppdeftot += data[year][str(opp)]['season'][1]
            reg_season_elo = game[-2]
        print team
        data[year][team]['season'].append(data[year][team]['season'][0]/(oppdeftot/float(len(sche))))
        data[year][team]['season'].append(data[year][team]['season'][1]/(oppofftot / float(len(sche))))
        data[year][team]['season'].append(reg_season_elo)

    # print teams
    # print years

    # stats = []
    # for team in teams:
    #      stats.append([team, data[year][team]['season'][2],data[year][team]['season'][3]])
    #
    # stats = np.array(stats)
    # print stats[np.argsort(stats[:,0])]
    # # print data[year]

# This is where the model training begins
trainyearstart = 2010
trainyearend = 2013
TS2003 = GT.GetTourneySche(trainyearstart,trainyearend)

for game in TS2003:
    team1 = game[0]
    team2 = game[1]
    trainyear = str(game[-1])
    game[0] = data[trainyear][team1]['season'][2]-data[trainyear][team2]['season'][2]
    game[1] = data[trainyear][team2]['season'][3] - data[trainyear][team1]['season'][3]
    game.insert(-2, data[trainyear][team1]['season'][-1] - data[trainyear][team2]['season'][-1])
TS2003 = np.array(TS2003)
TS2003 = TS2003[:,:-1]


testyearstart = 2014
testyearend= None
TS2004 = GT.GetTourneySche(testyearstart, testyearend, games=36)
test= np.array(TS2004)
print np.array(TS2004)
test_games=[]
for game in TS2004:
    team1 = game[0]
    team2 = game[1]
    test_games.append([team1,team2])

    testyear = str(game[-1])
    game[0] = data[testyear][team1]['season'][2]-data[testyear][team2]['season'][2]
    game[1] = data[testyear][team2]['season'][3] - data[testyear][team1]['season'][3]
    game.insert(-2,data[testyear][team1]['season'][-1]-data[testyear][team2]['season'][-1])

TS2004 = np.array(TS2004)
TS2004 = TS2004[:,:-1]
print TS2003.shape

model1 = MLPClassifier(hidden_layer_sizes=(100,100))
model2 = KNeighborsClassifier(n_neighbors=5,weights='uniform')
model3 = DecisionTreeClassifier()
model4 = SVC(probability=True)
model5 = AdaBoostClassifier()
vmodel = VotingClassifier(estimators=[('NN', model1), ('KNN', model2), ('DT', model3), ('SVC', model4), ('ADA', model5)], voting='soft')
# vmodel = LogisticRegression()
Scaler = StandardScaler()
TS2003x=Scaler.fit_transform(TS2003[:,:-1])
TS2004x = Scaler.transform(TS2004[:,:-1])
vmodel.fit(TS2003[:,:-1],TS2003[:,-1])

predicted = vmodel.predict(TS2004[:,:-1])
print predicted
predicted = np.reshape(predicted,(predicted.shape[0],1))
np.set_printoptions(linewidth=100)
check =np.append(TS2004,predicted,axis=1)
check = np.append(check,test[:,:2],axis=1)
print vmodel.classes_
print check
# if actual is 0, that means team 1 won
Seeds = GT.Seedings()

for i,game in enumerate(test_games):
    if check[i][3] == "0.0":
        if check[i][2] == "0.0":
            print str(Seeds[str(testyearstart)][game[0]]) + ")"+ game[0] + " beats " +str(Seeds[str(testyearstart)][game[1]]) + ")"+ game[1] + ": right"
        else:
            print str(Seeds[str(testyearstart)][game[0]]) + ")" + game[0] + " beats " + str(Seeds[str(testyearstart)][game[1]]) + ")" + game[1] + ": wrong"

    if check[i][3] == "1.0":
        if check[i][2] == "1.0":
            print str(Seeds[str(testyearstart)][game[1]]) + ")" + game[1] + " beats " + str(
                Seeds[str(testyearstart)][game[0]]) + ")" + game[0] + ": right"
        else:
            print str(Seeds[str(testyearstart)][game[1]]) + ")" + game[1] + " beats " + str(
                Seeds[str(testyearstart)][game[0]]) + ")" + game[0] + ": wrong"

print confusion_matrix(TS2004[:,-1],predicted)

# plt.plot(TS2003[:,-1],TS2003[:,0]-TS2003[:,1], 'bo')
# plt.show()