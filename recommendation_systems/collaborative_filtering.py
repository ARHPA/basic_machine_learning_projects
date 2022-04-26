
import numpy as np
import pandas as pd

movieDf = pd.read_csv("movies.csv")
ratingDf = pd.read_csv("ratings.csv")

movieDf["year"] = movieDf["title"].str.extract('(\(\d\d\d\d\))')
movieDf["year"] = movieDf["year"].str.extract('(\d\d\d\d)')
movieDf['year'] = pd.to_numeric(movieDf['year'])
movieDf["title"] = movieDf["title"].str.replace('(\(\d\d\d\d\))', '').apply(lambda x: x.strip())
movieDf = movieDf.drop("genres", axis=1)
ratingDf = ratingDf.drop("timestamp", axis=1)

inputDf = [
            {'title':'Iron Man', 'rating':4.5},
            {'title':'Interstellar', 'rating':5},
            {'title':'Captain America: The First Avenger', 'rating':4.5},
            {'title':'Iron Man 3', 'rating':4},
            {'title':'Thor', 'rating':4},
            {'title':"Pulp Fiction", 'rating':3.5},
         ] 
inputDf = pd.DataFrame(inputDf)
inputMovie = movieDf[movieDf["title"].isin(inputDf["title"].tolist())]
inputMovie = inputMovie.merge(inputDf).drop("title", axis=1)

inputRating = ratingDf[ratingDf["movieId"].isin(inputMovie["movieId"].tolist())]
userGroup = inputRating.groupby(["userId"])
userGroup = sorted(userGroup, key=lambda x: len(x[1]), reverse=True)
userGroup = userGroup[0:100]

personDic = {}

for id, group in userGroup:
    inputMovieList = inputMovie[inputMovie["movieId"].isin(group["movieId"].tolist())]
    inputMovieList = inputMovieList.sort_values(by="movieId")
    group = group.sort_values(by="movieId")
    inputRatingList = inputMovieList["rating"]
    groupRating = group["rating"]
    nRatings = len(group)
    xbar = pow(sum(inputRatingList),2)/float(nRatings)
    ybar = pow(sum(groupRating),2)/float(nRatings)
    xx = sum(x**2 for x in inputRatingList) - xbar
    yy = sum(y**2 for y in groupRating) - ybar
    xy = sum(x*y for x, y in zip(inputRatingList, groupRating)) - sum(inputRatingList)*sum(groupRating)/float(nRatings)
    if xx and yy:
        personDic[id] = xy / np.sqrt(xx * yy)
    else:
        personDic[id] = 0

personSim = pd.DataFrame.from_dict(personDic, orient='index')
personSim.columns = ["similarity"]
personSim["userId"] = personSim.index
personSim.index = range(len(personSim))
personSim = personSim.sort_values(by="similarity", ascending=False)[:50]

allMovieRating = personSim.merge(ratingDf)
allMovieRating['weightedRating'] = allMovieRating['similarity'] * allMovieRating["rating"]

allMovieGroup = allMovieRating.groupby("movieId").sum()[['weightedRating']]
allMovieGroup.columns = ['sum of ratings']

recommendationDf = pd.DataFrame()
recommendationDf['score'] = allMovieGroup["sum of ratings"] / len(allMovieGroup)
recommendationDf['movieId'] = allMovieGroup.index
recommendationDf = recommendationDf.sort_values(by='score',ascending=False)
recommendationDf.index = range(len(recommendationDf))
recommendationDf = recommendationDf.merge(movieDf)

recommendationDf = recommendationDf[~recommendationDf['movieId'].isin(inputMovie['movieId'])]
#recommendationDf = recommendationDf[recommendationDf["year"] >= 1970]

print("\n\n20 best movies you should to see:")
print(recommendationDf.head(20))