import pandas as pd

team_names = pd.read_csv('Teams.csv')
team_names = team_names.set_index('Team_Id').to_dict()
print team_names

data = pd.read_csv('RegularSeasonDetailedResults.csv')
print data.head()

teams = data['Wteam'].unique()

for team in teams:
    season = data[data['Season']==2016]
    W=season[data['Wteam'] == 1104]
    Wmean=W.mean(numeric_only=True)
    Wopp = W['Lteam'].unique()
    Wcount = W.shape[0]
    print Wmean['Wscore']

    for opp in Wopp:


    L=data[data['Lteam']==1104].sum(numeric_only=True)



data = data[(data['Wteam']==1104)| (data['Lteam']==1104)]
print data[data['Season']==2016]
