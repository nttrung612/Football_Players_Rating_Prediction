import pandas as pd

df = pd.read_csv('fifa_players.csv')

df.drop(['full_name', 'birth_date'], axis=1, inplace=True)

df.drop(['national_team', 'national_team_position','national_rating', 
         'national_jersey_number'], axis=1, inplace=True)

valid_body_types = ['Lean', 'Normal', 'Stocky']
df = df[df['body_type'].isin(valid_body_types)]

df.drop(['international_reputation(1-5)'], axis=1, inplace=True)

df.drop(['value_euro', 'wage_euro', 'release_clause_euro',
         'potential', 'composure', 'reactions'], axis=1, inplace=True)

df.to_csv('full_data.csv', index=False)