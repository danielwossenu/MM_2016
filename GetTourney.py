import csv
import numpy as np

def Seedings():
    Seeds = {}
    filereaderSeeds = csv.reader(open("SeedingsCleaned.csv"), delimiter=",")
    header = filereaderSeeds.next()
    for seed in filereaderSeeds:
        if seed[0] not in Seeds:
            Seeds[seed[0]] = {}
        Seeds[seed[0]][seed[2]] = seed[1]
    return Seeds


def RealSeedings():
    Seeds = {}
    filereaderSeeds = csv.reader(open("TourneySeeds.csv"), delimiter=",")
    header = filereaderSeeds.next()
    for seed in filereaderSeeds:
        if seed[0] not in Seeds:
            Seeds[seed[0]] = {}
        Seeds[seed[0]][seed[1]] = seed[2]
    return Seeds


def GetTourneySche_train(startyear,endyear=None,games=64):
    TS = []
    Seeds=Seedings()
    filereaderSeeds = csv.reader(open("SeedingsCleaned.csv"), delimiter=",")
    header = filereaderSeeds.next()
    for seed in filereaderSeeds:
        if seed[0] not in Seeds:
            Seeds[seed[0]]={}
        Seeds[seed[0]][seed[2]]=seed[1]
    switch = -1

    # while t < games:
    if endyear==None:
        endyear = startyear
    for year in range(startyear,endyear+1):
        t=0
        filereader = csv.reader(open("TourneyCompactResults.csv"), delimiter=",")
        header = filereader.next()
        for game in filereader:
            if t < games:
                if game[0] == str(year):
                    #switch is randomly 1 or -1 so that it randomized whether the 1st team wins or the 2nd team wins
                    # this creates equal classes for the model to train on
                    if switch == -1:
                        a = Seeds[str(year)][game[2]]
                        b = Seeds[str(year)][game[4]]
                        # TS.append([game[2], game[4],int(Seeds[str(year)][game[2]])-int(Seeds[str(year)][game[4]]), 0,year])
                        # this code below creates tourney data w/o seeds for each team
                        TS.append([game[2], game[4], 0, year])
                        TS.append([game[4], game[2], 1, year])
                    if switch == 1:
                        # TS.append([game[4], game[2],int(Seeds[str(year)][game[4]])-int(Seeds[str(year)][game[2]]), 1, year])
                        # this code below creates tourney data w/o seeds for each team
                        TS.append([game[4], game[2], 1, year])
                        TS.append([game[2], game[4], 0, year])
                    switch *= np.random.choice([-1,1])
                    t+=1
    return TS


def GetTourneySche_test(startyear,endyear=None,games=64):
    TS = []
    Seeds=Seedings()
    filereaderSeeds = csv.reader(open("SeedingsCleaned.csv"), delimiter=",")
    header = filereaderSeeds.next()
    for seed in filereaderSeeds:
        if seed[0] not in Seeds:
            Seeds[seed[0]]={}
        Seeds[seed[0]][seed[2]]=seed[1]
    switch = -1

    # while t < games:
    if endyear==None:
        endyear = startyear
    for year in range(startyear,endyear+1):
        t=0
        filereader = csv.reader(open("TourneyCompactResults.csv"), delimiter=",")
        header = filereader.next()
        for game in filereader:
            if t < games:
                if game[0] == str(year):
                    #switch is randomly 1 or -1 so that it randomized whether the 1st team wins or the 2nd team wins
                    # this creates equal classes for the model to train on
                    if switch == -1:
                        a = Seeds[str(year)][game[2]]
                        b = Seeds[str(year)][game[4]]
                        # TS.append([game[2], game[4],int(Seeds[str(year)][game[2]])-int(Seeds[str(year)][game[4]]), 0,year])
                        # this code below creates tourney data w/o seeds for each team
                        TS.append([game[2], game[4], 0, year])
                    if switch == 1:
                        # TS.append([game[4], game[2],int(Seeds[str(year)][game[4]])-int(Seeds[str(year)][game[2]]), 1, year])
                        # this code below creates tourney data w/o seeds for each team
                        TS.append([game[4], game[2], 1, year])
                    switch *= np.random.choice([-1,1])
                    t+=1
    return TS


def GetSlots(year):
    Slots=[]
    filereaderSlots = csv.reader(open("TourneySlots.csv"), delimiter=",")
    header = filereaderSlots.next()
    for seed in filereaderSlots:
        if seed[0] == str(year):
            Slots.append(seed[1:])
    return Slots

def GetTeams():
    TEAMS={}
    filereaderSlots = csv.reader(open("Teams.csv"), delimiter=",")
    header = filereaderSlots.next()
    for team in filereaderSlots:
        TEAMS[team[0]]=team[1]
    return TEAMS

# print GetTourneySche(2004,games=2)


