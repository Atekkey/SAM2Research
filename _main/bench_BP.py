# To use: Remove Global scores line from the results.csv files


import pandas as pd
import numpy as np
dfSam = pd.read_csv("/work/hdd/bdnb/atekkey/sam2/notebooks/results/LVOS/results.csv") 
dfBP = pd.read_csv("/work/hdd/bdnb/atekkey/sam2/notebooks/results/BP_reverse/results.csv")
#/projects/bdnb/dzhao3/outputs/score_cache/results.csv


dfBP.columns = dfBP.columns.str.replace(' ', '')
dfSam.columns = dfSam.columns.str.replace(' ', '')

dfSam = dfSam.sort_values(by=['sequence'])
dfSam = dfSam.reset_index(drop=True)
dfBP = dfBP.sort_values(by=['sequence'])
dfBP = dfBP.reset_index(drop=True)
###

print("\n\n\n")


dfDiff = dfBP.drop(columns=["sequence"]) - dfSam.drop(columns=["sequence"]) 

idMap = {}
for i in range(dfBP.shape[0]):
    idMap[i] = dfBP["sequence"][i]

dfComp = pd.DataFrame()
dfComp["JF1"] = dfBP["J&F"]
dfComp["JF2"] = dfSam["J&F"]

dfComp['max'] = dfComp[['JF1', 'JF2']].max(axis=1)

print(dfComp.mean())


print("\n")
print("Difference stats (means):\n \t j    f       j&f") # Positive is BP is better
print("AVG: ", round(dfDiff["J"].mean(), 4), round(dfDiff["F"].mean(), 4), round(dfDiff["J&F"].mean(), 4))
comp = "better" if round(dfDiff["J&F"].mean(), 4) > 0 else "worse" 
print(f"On average it did {comp} than sam2")
###
print("\n")
dfBool = dfDiff["J"] >= 0
# false, true = dfBool.value_counts().iloc[1], dfBool.value_counts().iloc[0]
# print(f"BP did better: {true}, Sam did better: {false} out of {dfBool.shape[0]}")
v = (dfBool.value_counts())

print("# of bettered videos: ", str(int(v[True])) )
print("# of worsened videos: ", str(int(v[False])) )

###
print("\n")
noFold = dfDiff.copy()
print(f"If BP increased accuracy, it did so by an average of \n{noFold[noFold > 0].mean()}")
print(f"If BP decreased accuracy, it did so by an average of \n{-noFold[noFold < 0].mean()}")
###
print("\n")


def goodChanges(dfJ, idMap, threshhold = 15, verbose = True):
    count = 0
    for i, j in enumerate(dfJ):
        j = round(j, 2)
        if(j > threshhold):
            count += 1
            if(verbose):
                print(idMap[i], " +", j, "%")
    print("Count: ", count)

# Videos of note:
def badChanges(dfJ, idMap, threshhold = 15, verbose = True):
    count = 0
    for i, j in enumerate(dfJ):
        j = round(j, 2)
        if(j < -1 * threshhold):
            count += 1
            if(verbose):
                print(idMap[i], j, "%")
    print("Count: ", count)

print("Good changes over 5%")
goodChanges(dfDiff["J&F"],  idMap, threshhold = 5)
print("Bad changes under 0%")
badChanges(dfDiff["J&F"], idMap, threshhold = 0)
