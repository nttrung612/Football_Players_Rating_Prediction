# Football Players Rating Prediction

## Description
This project utilizes a comprehensive dataset containing detailed information on approximately 17,000 FIFA football players, meticulously scraped from SoFIFA.com. The dataset includes a wide range of player-specific attributes such as player names, nationalities, clubs, ratings, positions, skill attributes, and much more.

The primary goal of this project is to develop a machine learning model that predicts the overall rating of football players based on their various features, including skill levels, market value, potential, and other relevant characteristics.

## Dataset Features
The dataset contains the following features:
- **name**: Name of the player
- **full_name**: Full name of the player
- **birth_date**: Date of birth of the player
- **age**: Age of the player
- **height_cm**: Player's height in centimeters
- **weight_kgs**: Player's weight in kilograms
- **positions**: Positions the player can play
- **nationality**: Player's nationality
- **overall_rating**: Overall rating of the player in FIFA
- **potential**: Potential rating of the player in FIFA
- **value_euro**: Market value of the player in euros
- **wage_euro**: Weekly wage of the player in euros
- **preferred_foot**: Player's preferred foot
- **international_reputation(1-5)**: International reputation rating from 1 to 5
- **weak_foot(1-5)**: Rating of the player's weaker foot from 1 to 5
- **skill_moves(1-5)**: Skill moves rating from 1 to 5
- And many more features, including detailed skill attributes like **crossing**, **finishing**, **dribbling**, **strength**, etc.

## Use Cases
The dataset and model can be used for:
- **Player Performance Analysis**: Analyze and compare the performance of players based on their ratings and attributes.
- **Market Value Assessment**: Predict and assess the market value and wages of football players.
- **Team Composition and Strategy Planning**: Understand player capabilities to build effective team strategies.
- **Machine Learning Models**: Develop predictive models to estimate player potential and future career trajectories.

## Project Objectives
- **Data Preprocessing**: Handle missing values, categorical data, and data normalization to prepare the dataset for modeling.
- **Exploratory Data Analysis (EDA)**: Discover relationships between different features and understand the key factors influencing a player's overall rating.
- **Model Selection**: Train and evaluate various machine learning models to find the best model for predicting player ratings.
- **Hyperparameter Tuning**: Optimize the model's hyperparameters to achieve the best performance.

## Tools and Technologies Used
- **Python**: For data analysis and modeling.
- **Pandas & NumPy**: For data manipulation and preprocessing.
- **Matplotlib & Seaborn**: For data visualization and exploratory analysis.
- **Scikit-Learn**: For machine learning modeling, training, and evaluation.

## Important Note
This dataset is intended for educational and research purposes only. Please ensure to adhere to the terms of service of SoFIFA.com and relevant data protection laws when using this dataset. It should not be used for commercial purposes without proper authorization.
