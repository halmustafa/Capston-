# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
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

#exploring the data :

edx %>% tibble()
edx %>% summarise(
  uniq_movies = n_distinct(movieId),
  uniq_users = n_distinct(userId),
  uniq_genres = n_distinct(genres))


#sparse the entire matrix ,
#for a random sample of 100 movies and 100 users.
users <- sample(unique(edx$userId), 100)
rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")


#split  edx in  the training  and  test set.
set.seed(1)
test_index <-createDataPartition(y = edx$rating, times = 1,
                                 p = 0.1, list = FALSE)
edx_train <-edx[-test_index,]
edx_temp <-edx[test_index,]

edx_test <-edx_temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
#Add the Rows removed from the edx_test back into edx_train
removed <-anti_join(edx_temp, edx_test)
edx_train <-rbind(edx_train, removed)
rm(edx_temp, test_index, removed)


# Creat and train the algorithm:

#Loss function:

#compare different models,
#and define RMSE -residual mean squared error-
RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))
}





#model_1 (Just the average) 
#take a look at the rating distribution.
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill="yellow") + 
  scale_x_log10() + 
  ggtitle("Movies")

#some movies get rated more than others.

#comput averag rating mu :
mu_hat <- mean(edx_train$rating)
mu_hat
#know compute the residual mean squared error on 
#the test set data.
naive_rmse <- RMSE(edx_test$rating, mu_hat)
naive_rmse
# store the results in tibble
rmse_results <- tibble (method = "Just the average", RMSE = naive_rmse)
rmse_results

#model_2 (Movie Effect)
#we see that some movies get rated more than others.
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill="yellow") + 
  scale_x_log10() + 
  ggtitle("Movies")

mu <- mean(edx_train$rating) 
#compute b-i and data cleaning
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
rmse_m_2 <- RMSE(predicted_ratings, edx_test$rating)
rmse_m_2
rmse_results<- bind_rows(rmse_results,tibble(method="movie Effect model",
                                             RMSE=rmse_m_2 )) 
rmse_results%>% knitr::kable()

#model_3 (user-specific effect)
#from this plot we see that different users 
#different in terms of how they rate movies
edx_train %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black" ,fill= "yellow")

user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
#predicting values and computing the residual mean squared error.
predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
rmse_m_3 <- RMSE(predicted_ratings, edx_test$rating)
rmse_m_3
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                 RMSE = rmse_m_3 ))
rmse_results %>% knitr::kable()

#model_4 (Regularization)
#Here are 10 of the largest mistakes that we made when only
# using the movie effects in our models.
edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  select(title,  residual) %>% slice(1:10) %>% knitr::kable()
## compute these regularized estimates of b_i using lambda equals to 3.0 :
lambda <- 3
mu <- mean(edx_train$rating)
movie_reg_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

# plot of the regularized estimate versus the least square estimates 
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)


predicted_ratings <- edx_test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

#we are going to use cross-validation to choose lambda.

lambdas <- seq(0, 10, 0.5)

mu <- mean(edx_train$rating)
just_the_sum <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- edx_test %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test$rating))
})
qplot(lambdas, rmses) 

lambdas[which.min(rmses)]

#We can also use regularization to estimate the user effect.

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


rmse_results <- bind_rows(rmse_results,
                          tibble( method ="Regularized Movie Effect Model",  
                                  RMSE = min(rmses) ))
rmse_results %>% knitr::kable()

