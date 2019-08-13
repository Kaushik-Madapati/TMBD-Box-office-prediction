# TMBD-Box-office-prediction
Problem : 
             TMDB Box office Prediction challenge has  7,000 past films from The Movie Database to try and predict their overall 
             worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters,
             release dates, languages, production companies, and countries.

## 1. Data processing : ##

### 1.1 Missing Data : ###
        Using pandas info() find the missing data . Once we identify which feature has misisng data, it will be dealt differently for different features 
        
| Feature   |   Action                    |
| --------- | --------------------------- |
| belongs_to collection | Replaced missing data with “No data” while passing Json |
| genres | Replaced missing data with “No data” while passing Json |
| homepage | No Action taken in data processing. This features was dropped while selecting feature in feature engineering |
| Overview |No Action taken in data processing. This features was dropped while selecting features in feature engineering |
| poster_path |No Action taken in data processing. This features was dropped while selecting features in feature engineering |
| production_countries |Replaced missing data with “No data” while passing Json |
| spoken_languages |Replaced missing data with “No data” while passing Json |
| tagline |No Action taken in data processing. This features was dropped while selecting features in feature engineering |
| Keywords |Replaced missing data with “No data” while passing Json |
| cast |Replaced missing data with “No data” while passing Json |
| crew |Replaced missing data with “No data” while passing Json |

### 1.2. Extracting data from Json : ###
         1. All the Json data with  “‘“ replace with “ “ “  before passing and all missing data is replaced 
         with “No data”
         2. Once Json object are converting into strings. Next step is to extract only required information
         from these string because these string has lot of information which we don’t need.
### 1.3. One-hot encoding : ###
        Once we extract required information, the next step is to convert this list into one-hot encoding. 
### 1.4. Normalization : ###
        budget and revenue value ranges  are too high, to get better results we need to normalization  the data
        
## 2. Feature Engineering : ##
### 2.1 Selecting required features ### 
   Once we have all the data in a format where our model can understand, next step, we need to decide which of these 
   features are useful and also figure out what new features need to created using exisitng once. 
   
| Feature   |   Reason for dropping |
| --------- | --------------------------- |
| belongs_to collection | Not enough data |
| homepage |It's’ hard to extract relevant information from this string  |
| genres | Genres  are converted into one-hard encoding .  |
| imdb_id |This information is relevant for problem we are trying to solve |
| production_companies |Production Companies are converted into one-hard encoding |
| production_countries |This information is relevant for problem we are trying to solve|
| original_title |It's’ hard to extract relevant information from this string|
| overview | It's’ hard to extract relevant information from this string |
| poster_path | This information is relevant for problem we are trying to solve |
| spoken_languages |Spoken Languages  are converted into one-hard encoding  |
| Keywords | Keywords  are converted into one-hard encoding |
| status | This information is relevant for problem we are trying to solve |
| tagline |It's’ hard to extract relevant information from this string|
| title | It's’ hard to extract relevant information from this string |
| Keywords |Replaced missing data with “No data” while passing Json |
| cast |Cast are converted into one-hard encoding |
| crew |Crew are converted into one-hard encoding | 
| release_date |Release month is  converted into one-hard encoding |  
| original_language |This information is relevant for problem we are trying to solve|  
| id |This information is relevant for problem we are trying to solve|  
### 2.2 PCA ### 
  After converting some of the features into one-hot encoded, total no. of features went up 1500 which is too high.
  To reduce no. of  features, I used PCA on  original features 
## 3. Model ##
### 3.1 Train and Test data ###
   Split data into train and test Using sklearn train_test_split()
### 3.2 Model selection ###
    After evaluating different regression models by comparing their R^2
| Model |  R square value |
| ----- | --------------- |
|LinearRegression | 0.579235718330101 |
|Lasso | 0.6621628857326273 |
|ElasticNet | 0.0003453312418009169 |
|Ridge | 0.5431435846082412 |

After checking  R^2 of different model . I decided to narrow down to 2 model Linear regression and Lasso. 

Linear vs Lasso :
* Linear regression  in my case tend to overfitting . R^2 value looks good for train data but for test data 
  predicted values are not good 
* Lasso predicted values from test data are better compared to Linear regression
* Lasso also help to reduce the no. of features. 

### 3.3 Final model ###
Final model I used was Lasso.


 



