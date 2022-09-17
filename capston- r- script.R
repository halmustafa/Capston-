# Create edx set, validation set (final hold-out test set)
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)

# MovieLens 10M dataset:

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Data exploration :
  edx %>% as_tibble()
#We can see this table is in tidy format with thousands of rows,
#Each row represents a rating given by one user to one movie.
  
  glimpse(edx)
#   There is 9,000,061 Row , aand 6 Columns.
  
 ## edx information:
   # We can see the number of unique users that provide ratings
 # and for how many unique movies they provided them
  
  edx %>%
    summarize(n_users = n_distinct(userId),
              n_movies = n_distinct(movieId))
  
  #we can see a very small subset of 5 user and 5 movie:
  
  
  keep <- edx %>%
    dplyr::count(movieId) %>%
    top_n(5) %>%
    pull(movieId)

  
  tab <- edx %>%
    filter(userId %in% c(34:38)) %>% 
    filter(movieId %in% keep) %>% 
    select(userId, title, rating) %>% 
    spread(title, rating)
  tab %>% knitr::kable()
  #we can see the ratings that each user gave each movie
  #and we also see NA’s for movies that they didn’t watch or they didn’t rate.
  
  
  
 # matrix sparse :
  
  users <- sample(unique(edx$userId), 100)
  rafalib::mypar()
  edx %>% filter(userId %in% users) %>% 
    select(userId, movieId, rating) %>%
    mutate(rating = 1) %>%
    spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
    as.matrix() %>% t(.) %>%
    image(1:100, 1:100,. , xlab="Movies", ylab="Users")
  abline(h=0:100+0.5, v=0:100+0.5, col = "yellow")

  
  
  #the Genres:
  
  edx %>% 
    summarize(n_genres = n_distinct(genres))

  
  ##   the general properties of the data:
  
  #The first thing we notice is that some movies get rated more than others.plot_1
  
  edx %>% 
    dplyr::count(movieId) %>% 
    ggplot(aes(n)) + 
    geom_histogram(bins = 30, color = "green",fill="yellow") + 
    scale_x_log10() + 
    ggtitle("Movies")

  #  A second observation is that some users are more active than others at rating movies
  
  edx %>%
    dplyr::count(userId) %>% 
    ggplot(aes(n)) + 
    geom_histogram(bins = 30, color = "green", fill="yellow") + 
    scale_x_log10() +
    ggtitle("Users")

  
  ##   split edx in tow sets and create train and test set
  
  library(caret)
  set.seed(1)
  test_index <- createDataPartition(y = edx$rating, times = 1,
                                    p = 0.2, list = FALSE)
  edx_train <- edx[-test_index,]
  edx_temp <- edx[test_index,]
  
  
  edx_test <- edx_temp %>% 
    semi_join(edx_train, by = "movieId") %>%
    semi_join(edx_train, by = "userId")

  
  #   Loss funktion and compare models:
  
  #To compare different models or to see how well we’re doing compared to some baseline
#  , we need to quantify what it means to do well. 
#  We need a loss function. 
#  we are going to use the residual mean squared error on a test set.
 # to To compare different models
  
  
  
  #Define RMSE
  
  
  RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }
  
  
  # Building the Recommendation System:
  
  
  #model_1 (Just the average):
  
  
 # we start by building the simplest possible recommendation system: we predict the same
#rating for all movies regardless of user and movie. this model would look something like this:
    
   ## Y-u,i =mu + epsilon-u,i##
  
  
#epsilon represents independent errors - mu represents the true rating for all movies and users.
# We know that the estimate that minimizes the residual mean squared error is the least
#squares estimate of mu. And in this case, that’s just the
#  average of all the ratings. So we’re predicting all 
# unknown ratings with this average.
  
  
  ###compute mu
  
  mu_hat <- mean(edx_train$rating)
  mu_hat

  
  naive_rmse <- RMSE(edx_test$rating, mu_hat)
  naive_rmse 

  #  So we get a residual mean squared error of about 1.
  
#RMSE table
  
  
 # Now because as we go along we will be comparing different
 # approaches,we’re going to create a table that’s going to store
 #the results that we obtain as we go along.
 # We’re going to call it RMSE results.
  
  
  rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)
  rmse_results

  
  
###   model_2 (Movie effect):
  
#We know that some movies are just generally rated higher than others.
#so We can augment our previous model by adding the term bi to represent
#  the average rating for movie i.
         ##  Y-u,i =mu + epsilon-u,i + b-i  ##
  
  
  
#because b-hat_i, is just the average of y_u,i minus the overall 
# mean for each movie, i. So we can compute them using this code
  

  
  mu <- mean(edx_train$rating) 
  movie_avgs <- edx_train %>% 
    group_by(movieId) %>% 
    summarize(b_i = mean(rating - mu))

  
  #   from this plot-3 we see that these estimates vary substantially ist
  
  qplot(b_i, data = movie_avgs, bins = 10, color = I("yellow"))

  
  #   Let’s see how much our prediction improves once we use
  
        ## y-u-i-hat = mu-hat + b-i-hat ##
  
  
  predicted_ratings <- mu +edx_test %>% 
    left_join(movie_avgs, by='movieId') %>%
    .$b_i

  
  
  model_1_rmse <- RMSE(predicted_ratings,edx_test$rating)
  
  rmse_results <- bind_rows(rmse_results,
                            tibble(method="MovieEffectModel",
                                   RMSE= model_1_rmse ))
  rmse_results %>% knitr::kable() 

  
###   model_3 (User Effects):
  
#different users are different in terms of how they rate movies 
  #  we compute the average rating for user u for those that have rated 100 or more movies plot_4:
  
  edx_train %>%
    group_by(userId) %>%
    filter(n()>=100) %>%
    summarize(b_u = mean(rating)) %>% 
    ggplot(aes(b_u)) + 
    geom_histogram(bins = 30, color = "green",fill="yellow")
  
#we can see that there is substantial variability across users as well
  
# This implies that a further improvement to our model may be:
  
    ##  Y-u,i =mu + epsilon-u,i + b-i + b-u ##
  
# We include a term, b_u, which is the user-specific effect.
# we will compute our approximation by computing the overall mean,
# u-hat, the movie effects, b-hat_i, and then estimating the user effects, b_u-hat,
# by taking the average of the residuals obtained after removing the overall mean and 
# the movie effect from the ratings y_u,i.
  
  user_avgs <- edx_train %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu - b_i))

  
  
  
  predicted_ratings <- edx_test %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred

  
  
  model_2_rmse <- RMSE(predicted_ratings, edx_test$rating)
  rmse_results <- bind_rows(rmse_results,
                            tibble(method="Movie + User Effects Model",  
                                   RMSE = model_2_rmse ))
  rmse_results %>% knitr::kable()

  
  
##   model_4 (Regularization):
  
  #our improvement in residual mean square error when we just included the movie effect was only about 5%.
  
  #to see what’s going on, let’s look at the top 10 best movies based on the estimates of the movie effect b-hat_i.
 
  
  
  set.seed(755)
  movie_titles <- edx %>% 
    select(movieId, title) %>%
    distinct()
  movie_avgs %>% left_join(movie_titles, by="movieId") %>%
    arrange(desc(b_i)) %>% 
    select(title, b_i) %>% 
    slice(1:10) %>%  
    knitr::kable()

  
  #   here are the best 10 movies according to our estimates
  
  
  #Here is the same table, but now we include the number of ratings
  
  
  edx_train %>% dplyr::count(movieId) %>% 
    left_join(movie_avgs) %>%
    left_join(movie_titles, by="movieId") %>%
    arrange(desc(b_i)) %>% 
    select(title, b_i, n) %>% 
    slice(1:10) %>% 
    knitr::kable()

  
  #  So the supposed best and worst movies were rated by very few users,
  #in most cases just one. These movies were mostly obscure ones.
  #This is because with just a few users, we have more uncertainty, 
  #therefore larger estimates of b-i, negative or positive, are more likely
  #when fewer users rate the movies.For this, we introduce the 
  #concept of regularization. Regularization permits us to 
  #penalize large estimates that come from small sample sizes.
  
  
  ## compute these regularized estimates of b_i using lambda equals to 3
  
  
  
  lambda <- 3
  
  
  mu <- mean(edx_train$rating)
  movie_reg_avgs <- edx_train %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
 
  
  #   here we can see how the estimates shrink plot5:
  
  tibble(original = movie_avgs$b_i, 
         regularlized = movie_reg_avgs$b_i, 
         n = movie_reg_avgs$n_i) %>%
    ggplot(aes(original, regularlized, size=sqrt(n))) + 
    geom_point(shape=1, alpha=0.5)

  
  
  
  #   know let’s look at our top 10 best movies based on the estimates we got when using regularization.
  
  
  edx_train %>%
    dplyr::count(movieId) %>% 
    left_join(movie_reg_avgs) %>%
    left_join(movie_titles, by="movieId") %>%
    arrange(desc(b_i)) %>% 
    select(title, b_i, n) %>% 
    slice(1:10) %>% 
    knitr::kable()

  
  
  
  
    #So do we improve our results?
  
  predicted_ratings <- edx_test %>% 
    left_join(movie_reg_avgs, by='movieId') %>%
    mutate(pred = mu + b_i) %>%
    .$pred

  
  model_3_rmse <- RMSE(predicted_ratings, edx_test$rating)
  rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Regularized Movie Effect Model",  
                                       RMSE = model_3_rmse ))
  
  
  
  rmse_results %>% knitr::kable()

  
  
  
  # lambda is a tuning parameter. We can use cross-validation to choose it.
  
  
  lambdas <- seq(0, 10, 0.5)
  mu <- mean (edx_train$rating)
  just_the_sum <- edx_train %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - mu), n_i = n())
  rmses <- sapply(lambdas, function(l){
    predicted_ratings <- edx_test %>% 
      left_join(just_the_sum, by='movieId') %>% 
      mutate(b_i = s/(n_i+l)) %>%
      mutate(pred = mu + b_i) %>%
      .$pred
    return(RMSE(predicted_ratings, edx_test$rating))
  })
  qplot(lambdas, rmses)  

  
  
  lambdas[which.min(rmses)]

  
  
  
  ##   useing regularization to estimate the user effect.
  
  #It includes the parameters for the user effects as well.
  
 # Here we again use cross-validation to pick lambda.
  
  
  
  lambdas <- seq(0, 10, 0.5)
  rmses <- sapply(lambdas, function(l){
    mu <- mean(edx_train$rating)
    b_i <- edx_train %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- edx_train %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    predicted_ratings <- 
      edx_test %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      .$pred
    return(RMSE(predicted_ratings, edx_test$rating))
  })
  
  qplot(lambdas, rmses)  

  
  
  
  
  lambda <- lambdas[which.min(rmses)]
  lambda

  
  
  
  
  
  rmse_results <- bind_rows(rmse_results,
                            tibble(method="Regularized Movie + User Effect Model",  
                                   RMSE = min(rmses)))
  rmse_results %>% knitr::kable()

  
  
  
  
   # now we applay all this an edx , validationset:
  
  
  
  
  
  
  lambdas <- seq(0, 10, 0.5)
  rmses <- sapply(lambdas, function(l){
    mu <- mean(edx $rating)
    b_i <- edx %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- edx %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    
    
    
    predicted_ratings <- 
      validation  %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      .$pred
    return(RMSE(predicted_ratings, validation$rating))
  })
  
  qplot(lambdas, rmses)  

  
  
  lambda <- lambdas[which.min(rmses)]
  lambda 

  
  
  
  
  rmse_results <- bind_rows(rmse_results,
                            tibble(method="Regularized Movie + User Effect Model",  
                                   RMSE = min(rmses)))
  rmse_results %>% knitr::kable()
     
  
  
  
################################################################
  