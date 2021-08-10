##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(randomForest)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

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
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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


# PROJECT

# Generate random Forest fit.
fit_rf <- randomForest(rating ~ ., data = edx)

# Predict the data obtained with the random Forest fit.
predict_rf <- predict(fit_rf, validation)

# Generate glm fit.
fit_glm <- train(rating ~ ., method = "glm", data = edx)

# Predict the data obtained with the glm fit.
predict_glm <- predict(fit_glm, validation)

# Generate knn fit.
fit_knn <- train(rating ~ ., method = "knn", data = edx)

# Predict the data obtained with the knn fit.
predict_knn <- predict(fit_knn, validation)




# Trying another approach, the one observed in the
# "recommendation system" video, to use as an ensemble with rf method.


# Making sure all test sets have no missing values.
test <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

validation <- validation %>%
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")


# Calculating the first term of the linear model: mu.
mu <- mean(edx$rating)

# Define lambdas for regularization.
lambdas <- seq(0, 10, 0.25)

# Create function to generate RMSEs.
rmses <- sapply(lambdas, function(l) {
  
  # Compute b_i term, the coefficient for the movie effect.
  movie_effects <- edx %>% group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + l)) 
  
  # Compute b_u term, the coefficient for the user effect.
  user_effects <- edx %>% left_join(movie_effects, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + l))
  
  # Compute the rating predictions in the validation set.
  test_pred <- validation %>% left_join(movie_effects, by = "movieId") %>% 
    left_join(user_effects, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  #Return the RMSE value between predictions and real values.
  return(RMSE(test_pred, validation$rating))
  
})


# Plot lambdas versus RMSE values.
qplot(lambdas, rmses)  

# Obtain the value of lambda that minimizes the RMSE.
lambdas[which.min(rmses)]

# Repeat the process using the ideal lambda.
lf <- 5.25

# Compute b_i term, the coefficient for the movie effect.
movie_effects <- edx %>% group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lf)) 

# Compute b_u term, the coefficient for the user effect.
user_effects <- edx %>% left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n() + lf))

# Compute the rating predictions in the test set.
test_pred <- validation %>% left_join(movie_effects, by = "movieId") %>% 
  left_join(user_effects, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Return the RMSE value between predictions and real values.
RMSE(test_pred, validation$rating)




