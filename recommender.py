import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading data from given csv files
users = pd.read_csv('users.csv')
courses = pd.read_csv('courses.csv')
interactions = pd.read_csv('interactions.csv)

# handling missing data
users['age'].fillna(users['age'].mean(), inplace=True)
interactions['rating'].fillna(0, inplace=True)

#normalizing engagement metrixs: watch_time
interactions['normalized_watch_time'] = interactions['watch_time'] / interactions['watch_time'].max()
interactions['normalized_ratings'] = interactions['rating'] / interactions['rating'].max()

plt.figure(figsize=(8,6))
sns.boxplot(x='category', y='ratings', data=courses)
plt.title("Rating distribution by category")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(interactions['watch_time'])
plt.title("Watch time distribution")
plt.show()


#popularity check based on average rating
popular_courses = courses.groupby('id')['ratings'].mean().sort_values(ascending=False)
print("\n Popular courses based on average rating: \n",popular_courses)

# Next steps:
# collaborative filtering for personalized recommendations

#popularity check based on watch time
popular_courses_watch_time = interactions.groupby('course_id')['watch_time'].sum().sort_values(ascending=False)
print("\n Popular courses based on total watch time: \n",popular_courses_watch_time)
