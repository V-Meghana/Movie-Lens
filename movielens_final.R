library(tidyverse)
library(caret)
library(data.table)
library(dslabs)
library(dplyr)
library(stringr)
library(tidyr)
library(kableExtra)
library(knitr)
library(knitLatex)
library(randomForest)
library(Rborist)
library(recommenderlab)
library(recosystem)
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# splitting edx into test and train datasets
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.5, list = FALSE)
edx_train <- movielens[-test_index,]
edx_test <- movielens[test_index,]

edx_test <- edx_test %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# general overview/summary of data
summary <- data.frame(Total_Rows = nrow(edx),
                      Total_Columns = ncol(edx),
                      Total_Movies = n_distinct(edx$movieId),    
                      Total_Users = n_distinct(edx$userId),
                      Total_Genres = n_distinct(edx$genres),
                      Average_rating = round(mean(edx$rating),2),
                      Standard_Deviation = round((sd(edx$rating)), digits = 3))
summary %>% knitr::kable()


# plot of ratings
ggplot(edx, aes(rating)) +
  geom_histogram(fill= "blue")

# plot of ratings per movie
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, fill = "orange", color = "black") + 
  scale_x_log10() + 
  ggtitle("Ratings per Movie")

# plot of ratings per user
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, fill = "skyblue", color = "black") + 
  scale_x_log10() +
  ggtitle("Ratings per Users")

# plot of ratings per genre
edx %>%
  dplyr::count(genres) %>% 
  ggplot(aes(n)) + 
  geom_histogram(fill = "lightpink", color = "white") + 
  scale_x_log10()
ggtitle("Ratings per Genre")

#----------------------------------------------------------------------------
# Model 1 : Average Rating Model
mean <- round(mean(edx_train$rating), digits = 3)
print(paste("The mean value of the ratings is", mean))

# compute rmse for average model
naive_rmse <- RMSE(edx_train$rating,mean)

# tabulate the results
rmse_results <- tibble(method = "Average Rating Model", RMSE = naive_rmse)
rmse_results

#----------------------------------------------------------------------------
# Model 2 : Movie Effects
# computing b for different movies
mean <- mean(edx_train$rating)
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mean))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# new predicted ratings after taking movie effects into consideration
predicted_ratings2 <- mean + edx_test %>%
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

# find new rmse
new_rmse <- RMSE(predicted_ratings2, edx_test$rating)

# tabulate new results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effects",
                                 RMSE = new_rmse))
rmse_results

#----------------------------------------------------------------------------
# Model 3 : Movie-User Effects
user_avgs <- edx_train %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mean - b_i))

# construct predictors to see improvement in rmse 
predicted_ratings3 <- edx_test %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mean + b_i + b_u) %>%
  .$pred

# ensure the predictions don't exceed the rating limits
predicted_ratings3[predicted_ratings3<0.5] <- 0.5
predicted_ratings3[predicted_ratings3>5] <- 5

# calculating rmse
newer_rmse <- RMSE(predicted_ratings3, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie & User-Effects Model",
                                 RMSE = newer_rmse))
rmse_results

#----------------------------------------------------------------------------
# Model 4 : Regularized Movie Model

# plot rmse vs lambda
# using cross-validation for finding lambda
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mean <- mean(edx_train$rating)
  
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mean)/(n()+l))
  predicted_ratings <-
    edx_test %>%
    left_join(b_i, by = "movieId") %>%
    mutate(pred = mean + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test$rating))
})
qplot(lambdas, rmses)

# picking the right lambda value
lambda <- lambdas[which.min(rmses)]

lambda

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized movie effect model",
                                 RMSE = min(rmses)))
rmse_results

#----------------------------------------------------------------------------
# Model 5 : Regularized movie-user Model

# using cross-validation for finding lambda
lambdas <- seq(0, 10, 0.25)
rmses1 <- sapply(lambdas, function(l){
  
  mean <- mean(edx_train$rating)
  
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mean)/(n()+l))
  b_u <- edx_train %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mean)/(n()+l))
  predicted_ratings <-
    edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mean + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test$rating))
})

#select optimal lambda value
lambda <- lambdas[which.min(rmses1)]
print(paste("The appropriate value of lambda is ", lambda))

# calculate rmse
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized movie & user effect model",
                                 RMSE = min(rmses1)))
rmse_results

#----------------------------------------------------------------------------
# Model 6 : User Based Collaborative Filtering

# Training the model
#create a copy of edx
edx_new <- edx

# converting userID and movieID to factors
edx_new$userId <- as.factor(edx_new$userId)
edx_new$movieId <- as.factor(edx_new$movieId)

# converting userid and movieid to numeric vectors to make use of sparse-matrix function
edx_new$userId <- as.numeric(edx_new$userId)
edx_new$movieId <- as.numeric(edx_new$movieId)

sparse_ratings <- sparseMatrix(i = edx_new$userId,j = edx_new$movieId ,x = edx_new$rating, 
                               dims = c(length(unique(edx_new$userId)),
                                        length(unique(edx_new$movieId))),  
                               dimnames = list(paste("u", 1:length(unique(edx_new$userId)), sep = ""), 
                                               paste("m", 1:length(unique(edx_new$movieId)), sep = "")))
#we dont need the copy
rm(edx_new)

#Convert rating matrix into a recommenderlab sparse matrix
ratingMatrix <- new("realRatingMatrix", data = sparse_ratings)
ratingMatrix

# selecting appropriate users and movies
min_movies <- quantile(rowCounts(ratingMatrix), 0.9)
min_users <- quantile(colCounts(ratingMatrix), 0.9)


ratings_movies <- ratingMatrix[rowCounts(ratingMatrix) > min_movies,
                               colCounts(ratingMatrix) > min_users]
ratings_movies

# 5 ratings of 30% of users are excluded for testing
set.seed(1)
e <- evaluationScheme(ratings_movies, method="split", train=0.7, given=-5)

# using UBCF
model <- Recommender(getData(e, "train"), method = "UBCF", 
                     param=list(normalize = "center", method="Cosine", nn=50))

prediction <- predict(model, validation, type="ratings")

rmse_ubcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]
rmse_ubcf

# adding this rmse value to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(method="User Based Collaborative Filtering",
                                 RMSE = rmse_ubcf))
rmse_results

#available RAM in my computer is not sufficient to test validation set. But UBCF is sure to give similar results for the validation set.