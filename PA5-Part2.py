
# coding: utf-8

# In[1]:

########### Do not change anything below

get_ipython().magic(u'matplotlib inline')

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame
from IPython.display import display

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns


import scipy as sp
#SVD for Sparse matrices
from scipy.sparse.linalg import svds

from sklearn.metrics.pairwise import euclidean_distances

try:
   import cPickle as pickle
except:
   import pickle

from collections import defaultdict, Counter
import operator

from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)

########################If needed you can import additional packages for helping you, although I would discourage it
########################Put all your imports in the space below. If you use some strange package, 
##########################provide clear information as to how we can install it.

#######################End imports###################################


#####Do not change anything below




# In[2]:

#Load the user data
users_df = pd.read_csv('ml-100k/u.user', sep='|', names=['UserId', 'Age', 'Gender', 'Occupation', 'ZipCode'])
#Load the movies data: we will only use movie id and title for this assignment
movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=['MovieId', 'Title'], usecols=range(2))
#Load the ratings data: ignore the timestamps
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['UserId', 'MovieId', 'Rating'],usecols=range(3))

#Working on three different data frames is a pain
# Let us create a single dataset by "joining" these three data frames
movie_ratings_df = pd.merge(movies_df, ratings_df)
movielens_df = pd.merge(movie_ratings_df, users_df)

movielens_df.head()


# In[4]:

import collections
#Task t2a: 
# Create a dictionary where key is Movie Name and value is id
#       You can either use the movies_df or read and parse the u.item file yourself
movie_name_to_id_dictionary = {}

movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=['MovieId', 'Title'], usecols=range(2))
movies_list = movies_df.values.tolist()
keys=[m[1] for m in movies_list]
values=[m[0] for m in movies_list]
print movies_list

movie_name_to_id_dictionary = collections.OrderedDict(zip(keys,values))
#***************************************************************************************************************************
#Write code to populate the movie names to this array
all_movie_names = keys
#print all_movie_names

#Your code below



# In[5]:

#Task t2b: Write a function that takes two inputs: 
#  movie_id: id of the movie and common_users: a set of user ids
# and returns the list of rows corresponding to their movie ratings 

def get_movie_reviews(movie_id, common_users):
    #Get a boolean vector for themovielens_dfns_dfs provided by users in common_users for movie movie_id
    # Hint: use the isin operator of Pandas
    mask = None
    
    #Create a subset of data where the mask is True i.e. only collect data from users who satisfy the condition above
    # Then sort them based on userid
    movie_ratings = None
    
    movie_ratings = movielens_df[(movielens_df.MovieId == movie_id) & (movielens_df.UserId.isin(list(common_users)))]
    #Do not change below
    #Return the unique set of ratings provided
    movie_ratings = movie_ratings[movie_ratings['UserId'].duplicated()==False]
    movie_ratings= movie_ratings.sort(['MovieId', 'UserId'] , ascending=[True, True])
    return movie_ratings


# In[6]:

#Do not change below

#Here are some sample test cases for evaluating t2b
print "get_movie_reviews(1, set([1]))"
display( get_movie_reviews(1, set([1])) )

print "get_movie_reviews(1, set(range(1, 10)))"
display( get_movie_reviews(1, set(range(1, 10))) )

print "get_movie_reviews(100, set(range(1, 10)))"
display( get_movie_reviews(100, set(range(1, 10))) )

print "get_movie_reviews(784, set(range(1, 784)))"
display( get_movie_reviews(784, set(range(1, 784))) )


# In[7]:

#Task t2c: Let us now calculate the similarity between two movies
# based on the set of users who rated both movies
#Using euclidean distance is a bad idea - but simplifies the code

def calculate_similarity(movie_name_1, movie_name_2, min_common_users=0):
    
    movie1 = movie_name_to_id_dictionary[movie_name_1]
    movie2 = movie_name_to_id_dictionary[movie_name_2]
    
    #This is the set of UNIQUE user ids  who reviewed  movie1
    users_who_rated_movie1 = set((movielens_df[(movielens_df.MovieId == movie1)].UserId).tolist())
    
    #This is the set of UNIQUE user ids  who reviewed  movie2
    users_who_rated_movie2 = set((movielens_df[(movielens_df.MovieId == movie2)].UserId).tolist())
    
    #Compute the common users who rated both movies: 
    # hint convert both to set and do the intersection
    common_users = users_who_rated_movie1.intersection(users_who_rated_movie2)
    
    #Using the code you wrote in t2a, get the reviews for the movies and common users
    movie1_reviews = get_movie_reviews(movie1, common_users)
    movie2_reviews = get_movie_reviews(movie2, common_users)
    
    #Now you have the data frame for both movies
    # Use the euclidean_distances function from sklean (imported already)
    # to compute the distance between their rating values
    distance = euclidean_distances(movie1_reviews['Rating'].values, movie2_reviews['Rating'].values)

    if len(common_users) < min_common_users:
        return [[float('inf')]]
    return distance

print calculate_similarity("Toy Story (1995)", "GoldenEye (1995)")
print calculate_similarity("GoldenEye (1995)", "Tomorrow Never Dies (1997)")
print calculate_similarity("Batman Forever (1995)", "Batman & Robin (1997)")


# In[8]:

#Task t2d: Given a movie, find the top-k most similar movies
# that have the lowest euclidean distance 

#Here is the high level logic:
#  for each movie in all_movie_names (Except the input movie name)
#    compute its similarity and store it in an array
#   return the k movies with the smallest distances
# remember to pass min_common_users to calculate_similarity
def get_top_k_similar_movies(input_movie_name, k=5, min_common_users=0):
    other_movies = [movie_name for movie_name in all_movie_names if movie_name != input_movie_name]
    similarity_list = []
    for movie in other_movies:
        similarity_list.append((calculate_similarity(input_movie_name, movie, min_common_users)[0][0], movie))
    
    similarity_list = sorted(similarity_list)
    
    return similarity_list[:k]


# In[9]:

#print get_top_k_similar_movies("Toy Story (1995)", 10)
print "\nMovies similar to GoldenEye [25]", get_top_k_similar_movies("GoldenEye (1995)", 10, 25)
print "\nMovies similar to GoldenEye [50]", get_top_k_similar_movies("GoldenEye (1995)", 10, 50)
print "\nMovies similar to GoldenEye [100]", get_top_k_similar_movies("GoldenEye (1995)", 10, 100)
print "\n\n"

print "\nMovies similar to Usual Suspects [25]", get_top_k_similar_movies("Usual Suspects, The (1995)", 10, 25)
print "\nMovies similar to Usual Suspects [50]", get_top_k_similar_movies("Usual Suspects, The (1995)", 10, 50)
print "\nMovies similar to Usual Suspects [100]", get_top_k_similar_movies("Usual Suspects, The (1995)", 10, 100)
print "\n\n"

print "\nMovies similar to Batman Forever [25]", get_top_k_similar_movies("Batman Forever (1995)", 10, 25)
print "\nMovies similar to Batman Forever [50]", get_top_k_similar_movies("Batman Forever (1995)", 10, 50)
print "\nMovies similar to Batman Forever [100]", get_top_k_similar_movies("Batman Forever (1995)", 10, 100)
print "\n\n"

print "\nMovies similar to Shawshank Redemption [25]", get_top_k_similar_movies("Shawshank Redemption, The (1994)", 10, 25)
print "\nMovies similar to Shawshank Redemption [50]", get_top_k_similar_movies("Shawshank Redemption, The (1994)", 10, 50)
print "\nMovies similar to Shawshank Redemption [100]", get_top_k_similar_movies("Shawshank Redemption, The (1994)", 10, 100)
print "\n\n"


# In[10]:

#################################TASK 3#############################################


# In[11]:

#Do not change below
# By default euclidean distance can be give arbitrary values
# Let us "normalize" it by limit its value between 0 and 1 and slightly change the interpretation
# 0 means that the preferences are very different
# 1 means that preferences are identical
# For tasks 3 and 4, remember to use this function

#Vec1 and vec2 are vectors
def euclidean_distance_normed(vec1, vec2):
    if len(vec1) == 0:
        return 0.0
    euc_distance = euclidean_distances(vec1, vec2)[0][0]
    return 1.0 / (1.0 + euc_distance)


# In[12]:

#Task t3a:
# In this task, you want to compute the similarity between two items
#  which in this case means ratings_df
# You can use code from 2c except that you must now call euclidean_distance_normed
#  when computing the distance

def calculate_similarity_normed(movie_name_1, movie_name_2, min_common_users=0):
    movie1 = movie_name_to_id_dictionary[movie_name_1]
    movie2 = movie_name_to_id_dictionary[movie_name_2]

    #This is the set of UNIQUE user ids  who reviewed  movie1
    users_who_rated_movie1 = set((movielens_df[(movielens_df.MovieId == movie1)].UserId).tolist())
    
    #This is the set of UNIQUE user ids  who reviewed  movie2
    users_who_rated_movie2 = set((movielens_df[(movielens_df.MovieId == movie2)].UserId).tolist())
    
    #Compute the common users who rated both movies: 
    # hint convert both to set and do the intersection
    common_users = users_who_rated_movie1.intersection(users_who_rated_movie2)
    
    #Using the code you wrote in t2a, get the reviews for the movies and common users
    movie1_reviews = get_movie_reviews(movie1, common_users)
    movie2_reviews = get_movie_reviews(movie2, common_users)
        
    #Do not change below
    
    #Now you have the data frame for both movies
    # Use the euclidean_distances function from sklean (imported already)
    # to compute the distance between their rating values
    distance = euclidean_distance_normed(movie1_reviews['Rating'].values, movie2_reviews['Rating'].values)

    if len(common_users) < min_common_users:
        return 0.0
    return distance


# In[13]:

#Do not change below
print calculate_similarity_normed("Toy Story (1995)", "GoldenEye (1995)")
print calculate_similarity_normed("GoldenEye (1995)", "Tomorrow Never Dies (1997)")
print calculate_similarity_normed("Batman Forever (1995)", "Batman & Robin (1997)")


# In[14]:

#Do not change below

#We are now going to create item-item similarity database
# Since our data is "small", we will use a non-traditional approach of using nested hashes
# In real-life, using something like databases or other data structures is far more preferable

#Here is the high level structure
#{
#    movie_name_1:  
#    { 
#        movie_name_2: similarity_between_movie_1_and_2, 
#        movie_name_3: similarity_between_movie_1_and_3, 
#        ....
#        movie_name_n: similarity_between_movie_1_and_n
#    },
#    movie_name_2:
#    {
#        movie_name_1: similarity_between_movie_2_and_1, 
#        movie_name_3: similarity_between_movie_2_and_3, 
#        ....
#        movie_name_n: similarity_between_movie_2_and_n
#    },
#    ....
#    movie_name_n:
#    {
#        movie_name_1: similarity_between_movie_n_and_1, 
#        movie_name_2: similarity_between_movie_n_and_2, 
#        ....
#        movie_name_n-1: similarity_between_movie_n_and_n-1
#    },
#}    
    

#Here is how to use this data structuere:

#To get similarity between movies
#   data[movie1][movie2]
#To get similarity between one movie and all others
#   data[movie1]


# In[15]:

#DO not change below
#This hash stores the movie to movie 
# as described above
movie_similarity_hash = defaultdict(dict)


# In[16]:



#Item based filtering is expensive as you need to compute similarity of all pairs of items
# for this dataset it is 1682*1682 ~ 28 lakh pairs or 2.8 million
# running all of them might take hours and close to a day
# instead let us run on a smaller dataset
# specifically, let us only focus on the top-250 movies based on ratings
# which is more manageable


#Task t3b: 
# Get the top-k movie names with most ratings
#Hint: use Counter class

def top_k_movie_names(k):
    movie_ratings_counter = Counter()
    
    #Your code below
    movie_ratings_counter.update(movielens_df['Title'].values)
    
    return movie_ratings_counter.most_common(k)


# In[17]:

#Do not change below
print "Top-10", top_k_movie_names(10), "\n"
print "Top-25", top_k_movie_names(25), "\n"


# In[18]:

#Do not change below
top_250_movie_names = [item[0] for item in top_k_movie_names(250)]


# In[19]:

#Task t3c:
#Use the following logic
#  for each movie in movie_names:
#    for all other movies in movie_names:
#      compute similarity between  two movies using calculate_similarity_normed
#      remember to pass min_common_users to that function
#  note that movie_similarity_hash is a defaultdict 
#  so similarity between movie1 and movie2 can be set as movie_similarity_hash[movie1][movie2]
#  btw, similarity in our case is commutative. 
#   i.e. similarity(movie1, movie2) = similarity(movie2, movie1)
#   so do not call the function twice !
# movie_names is an array that lists the movies for which you have to compute pairwise similarity
def compute_movie_to_movie_similarity(movie_names, min_common_users=0):
    #Your code below
    for movie in movie_names:
        other_movies = movie_names[movie_names.index(movie)+1:]
        for other_movie in other_movies:
            movie_similarity_hash[movie][other_movie] = calculate_similarity_normed(movie, other_movie, min_common_users)
            movie_similarity_hash[other_movie][movie] = movie_similarity_hash[movie][other_movie]
            


# In[20]:

#Do not change below

#Let us first test if your code above is correct by testing against a small subset of data
movie_similarity_hash = defaultdict(dict)
# let use the top-10 movies
compute_movie_to_movie_similarity(top_250_movie_names[:10], min_common_users=0)

#Get similarity with 
display(movie_similarity_hash["Toy Story (1995)"])
display(movie_similarity_hash['Return of the Jedi (1983)'])

print movie_similarity_hash["Toy Story (1995)"]["Independence Day (ID4) (1996)"]


# In[21]:

#Do not change below
#Let us now test against top-250 most popular movies
#This might take 10-20 mins to run!
movie_similarity_hash = defaultdict(dict)
compute_movie_to_movie_similarity(top_250_movie_names, min_common_users=25)


# In[23]:

#Do not change below
#Do this if you want to persist the data 

# Let us persist the movie-movie similarity data structure 
# that way you dont need to re-run the whole thing
#pickle is a serialization library in Python
# To persist/serialize, use the following line
pickle.dump(movie_similarity_hash, open("movie_similarity.pickle", "wb"))
# To deserialize, uncomment the following line 
#movie_similarity_hash = pickle.load( open( "movie_similarity.pickle", "rb" ) )


# In[24]:

#Do not change below
for movie_name in top_250_movie_names[:10]:
    print "Top-10 most similar movies for ", movie_name, " :", 
    print sorted(movie_similarity_hash[movie_name].items(), key=operator.itemgetter(1), reverse=True)[:10]
    print "\n"


# In[25]:

#Task t3d

#Before doing t3d, please complete t4a so that user_rating_hash is available
# this will make the code below easier

#In this task, we are going to predict the rating of a user u for a movie m using item based collaborative filtering
#Here is the high level logic:
# for each item i rated by this user:
#    s = similarity between i and input movie m 
#    if similarity between i and m is 0, ignore this item 
#    compute weighted rating for m based on i as rating for i * s
# compute the predicted rating as sum of all weighted ratings / sum of all similarities

def predict_rating_for_movie_icf(movie_similarity_hash, input_user_id, input_movie_name, movies_considered):
    total_weighted_rating = 0.0
    total_similarity= 0.0
        
    #Hint: movie_similarity_hash is a nested hash where user id is key and 
    #  all their rating as a dictionary  
    # this dictionary is ordered as moviename: rating

    #if this user has already rated the movie, return that rating
    if input_movie_name in user_rating_hash[input_user_id].keys():
        return user_rating_hash[input_user_id][input_movie_name]
    
        
    #For each movie the user has rated

        #if user rated some movie, but it is not in the subset of movies that we computed pairwise similarity
        # such as top-250, then do not consider it either
        # for this task, the input is in movies_considered 
        
        #compute similarity between movies
        #dont recompute = use the hash
        
        #Reject item if similarity is 0
                     
        #Compute weighted rating
        
        #update total_weighted_rating and total_similarity
        
    # user_rating_hash[308] will give us all the movies rated by user 308
    # user_rating_hash[308]['Birdcage, The (1996)'] will give us the rating for movie 'Birdcage, The (1996)' by user 308.
    # movie_similarity_hash[m1][m2] will give us the similarity between movies m1 and m2.
    
    for movie in user_rating_hash[input_user_id]:
        if movie not in movies_considered:
            continue
        else:
            similarity = movie_similarity_hash[input_movie_name][movie]
            if similarity == 0:
                pass
            else:
                weighted_rating = similarity * user_rating_hash[input_user_id][movie]
                total_weighted_rating += weighted_rating
                total_similarity += similarity
        
    #Do not change below
    if total_similarity == 0.0:
        return 0.0
    
    return total_weighted_rating / total_similarity


# In[28]:

#Do not change below
#Let us compute the rating for first 5 users for the top-20 movies that they have not seen
for user_id in range(1, 5+1):
    print user_id, [ (movie_name, 
                        round(predict_rating_for_movie_icf(movie_similarity_hash, user_id, movie_name, top_250_movie_names),2))
                       for movie_name in top_250_movie_names[:20] 
                        if movie_name not in user_rating_hash[user_id]]
           
#print movie_name, predict_rating_for_movie_icf(movie_similarity_hash, 1, 'Liar Liar (1997)', min_common_users=25)


# In[29]:

#Task t3e: 
#Here is the pseudocode for recommending movies
# for each movie this user has not rated in movies_considered:
#           predict rating for this movie and this user using t3d
#  return the top-k movies
def recommend_movies_icf(input_user_id, movies_considered, movie_similarity_hash,
                             user_rating_hash, k=10, min_common_movies=5):
    predicted_ratings = []
    
    #Your code here
    for movie in [m for m in movies_considered if m not in user_rating_hash[input_user_id]]:
        predicted_ratings.append((predict_rating_for_movie_icf(movie_similarity_hash, input_user_id, movie, top_250_movie_names), movie))

        
    return sorted(predicted_ratings, reverse=True)[:k]


# In[30]:

#Do not change below:

#Let us predict top-5 movies for first 10 users
for user_id in range(1,11):
    print user_id, recommend_movies_icf(user_id, top_250_movie_names, movie_similarity_hash, 
                               user_rating_hash, k=10, min_common_movies=5)


# In[32]:

#*****************************************************************************************************************************
#*************************************************************************************************************************
#                                      TASK 4 


# In[26]:

#Task t4a
#Create the data structure as discussed above
# here is the logic:
# for each line in file ml-100k/u.data:
#   set user_rating_hash[user][movie] = rating
# read the instructions above again!

def compute_user_rating_hash():
    user_rating_hash = defaultdict(dict)
    
    #Your code below    
    all_users = ratings_df.UserId.tolist()
    
    for user in all_users:
        movie_names = [all_movie_names[i-1] for i in ratings_df[(ratings_df.UserId == user)].MovieId.tolist()]
        ratings = ratings_df[(ratings_df.UserId == user)].Rating.tolist()
        user_rating_hash[user] = dict(zip(movie_names, ratings))
    
    return user_rating_hash


# In[27]:

#Do not change below
user_rating_hash = compute_user_rating_hash()


# In[28]:

#Do not change below
#How many users are there?
print len(user_rating_hash.keys())
#How many movies did each of the first 20 users rated?
print [len(user_rating_hash[i].keys()) for i in range(1,20+1)] 
#print the ratings of user 4
display(user_rating_hash[4])


# In[31]:

#Task t4b:
#We need to modify our logic for computing distance
#Here is the high level pseudocode:
# movie1 = movie names rated by user 1
# movie2 = movie names rated by user 2
# common movies = set intersection of movie1 and movie2
# if number of common movies is less than min_common_movies, return 0.0 [not 0]
# other wise create a vector with rating for common movies only
# compute euclidean distance between the vectors
# return 1 / (1+euclidean distace)

def compute_user_user_similarity(user_rating_hash, user_id_1, user_id_2, min_common_movies=0):
    #Get list of movie names rated by user 1. hint use keys function [see above for usage]
    
    movies_rated_by_user_1 = user_rating_hash[user_id_1].keys()
    movies_rated_by_user_2 = user_rating_hash[user_id_2].keys()
    
    #compute common movies
    common_movies = set(movies_rated_by_user_1).intersection(set(movies_rated_by_user_2))
    
    if len(common_movies) < min_common_movies:
        return 0.0
    
    common_movies = sorted(list(common_movies))
    
    #vector1 is the set of ratings for user1 for movies in common_movies
    vector1 = [user_rating_hash[user_id_1][movie] for movie in common_movies]
    #vector2 is the set of ratings for user2 for movies in common_movies
    vector2 = [user_rating_hash[user_id_2][movie] for movie in common_movies]
    
    #Compute distance and return 1.0/(1.0+distance)
    distance = euclidean_distances(vector1, vector2)[0][0]
    return 1.0 / ( 1.0 + distance)


# In[32]:

#Testing code
print [round(compute_user_user_similarity(user_rating_hash, 1, i),2) for i in range(1, 10+1)]
print [round(compute_user_user_similarity(user_rating_hash, 784, i),2) for i in range(1, 10+1)]


# In[33]:

#Task t4c
#This function finds the k-most similar users 
#Here is the high level logic:
#  for each user in all_user_ids other than the input user id:
#     find similarity between this user and input_user_id and store as (similarity, other userid)
#     sort based on similarity
#  return top-k
# remember to pass min_common_movies
def top_k_most_similar_users(user_rating_hash, input_user_id, all_user_ids, k=10, min_common_movies=0):
    user_similarity = []
        
    #Your code below
    for user in all_user_ids:
        if user != input_user_id:
            user_similarity.append((compute_user_user_similarity(user_rating_hash, input_user_id, user, min_common_movies), user))

    user_similarity = sorted(user_similarity, key=lambda x: x[0])
    
    return sorted(user_similarity, reverse=True)[:k]


# In[34]:

#Do not change below
all_user_ids = range(1, 943+1)
print top_k_most_similar_users(user_rating_hash, 1, all_user_ids, 10, 5)
print top_k_most_similar_users(user_rating_hash, 1, all_user_ids, 10, 10)
print top_k_most_similar_users(user_rating_hash, 812, all_user_ids, 10, 5)
print top_k_most_similar_users(user_rating_hash, 812, all_user_ids, 10, 20)


# In[62]:

#Task t4d
#In this task, we are going to predict the rating of a user for a movie using user based collaborative filtering
#Here is the high level logic:
# for each user u in all_user_ids:
#    s= similarity between u and input_user_id [remember to pass min_common_movies]
#    if similairty is 0.0 ignore u
#    if u has not rated this movie, ignore again
#    suppose u has rated this movie with a value of r
#    i am now going to give a "weighted rating" as r*s
# compute the predicted rating as sum of all weighted ratings / sum of all similarities

def predict_rating_for_movie_ucf(user_rating_hash, input_user_id, movie_name, all_user_ids, min_common_movies=5):
    total_weighted_rating = 0.0
    total_similarity= 0.0

    #For each user id
    for user in all_user_ids:
        #except input_user_id
        if user != input_user_id:
            #compute similarity between users
            similarity = compute_user_user_similarity(user_rating_hash, input_user_id, user, min_common_movies)
            #Reject user if similarity is 0
            if similarity != 0:
                #reject user if (s)he has not rated the movie
                if movie_name in user_rating_hash[user].keys():
                    #Compute weighted rating
                    r = user_rating_hash[user][movie_name]
                    weighted_rating = r * similarity
                    #update total_weighted_rating and total_similarity
                    total_weighted_rating += weighted_rating
                    total_similarity += similarity

    #Do not change below
    if total_similarity == 0.0:
        return 0.0
    
    return total_weighted_rating / total_similarity


# In[73]:

#Do not change below
all_user_ids = range(1, 943+1)
for user_id in range(1, 5+1):
    print "user_id = ", user_id
    print [ round(predict_rating_for_movie_ucf(user_rating_hash, user_id, all_movie_names[i], all_user_ids, min_common_movies=5),1)
          for i in range(1, 10+1)]
    print [ round(predict_rating_for_movie_ucf(user_rating_hash, user_id, all_movie_names[i], all_user_ids, min_common_movies=10),1)
          for i in range(1, 10+1)]
    print "\n"


# In[71]:

#Task t4e: 
#Here is the pseudocode for recommending movies
# for each movie this user has not rated:
#     for all other users:
#           predict rating for this movie and this user using t4d
#  return the top-k movies

def recommend_movies_ucf(user_rating_hash, all_user_ids, input_user_id, k=10, min_common_movies=5):
    predicted_ratings = []   
    
    #Your code here
    
    for movie in [m for m in all_movie_names if m not in user_rating_hash[input_user_id]]:
        predicted_ratings.append(( predict_rating_for_movie_ucf(user_rating_hash, input_user_id, movie, all_user_ids, min_common_movies),
                                          movie))
                
    return sorted(predicted_ratings, reverse=True)[:k]


# In[72]:

#Do not change below
all_user_ids = range(1, 943+1)

for user_id in range(1, 5):
    print recommend_movies_ucf(user_rating_hash, all_user_ids, user_id, 10, 5)


# In[ ]:



