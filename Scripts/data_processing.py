"""
Script for processing player data into suitable format for model training.

Adds in All-Star data into general player data
"""

import os
import unicodedata
import pandas as pd
import numpy as np
from scipy.stats import zscore

def combine_rosters():
    load_dir = 'raw_data'

    save_dir = "processed_data"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for year in range(1979, 2025):
        files = [f for f in os.listdir(load_dir) if f.startswith(str(year)) and f.endswith('.csv')]
        
        dfs = []

        # Loop through files
        for file in files:
            file_path = os.path.join(load_dir, file)  
            df = pd.read_csv(file_path)  
            df = df.drop(columns=['All Star', 'Year'])
            dfs.append(df) 

        combined_df = pd.concat(dfs, ignore_index=True)

        # Bad data for 2021, remove specific row to prevent compounding error
        if (year == 2021):
            combined_df = combined_df.drop(232)
        
        output_file = os.path.join(save_dir, f'{year}_player_stats.csv') 
        combined_df.to_csv(output_file, index=False)

def combine_seasons():
    all_players = pd.DataFrame()

    for year in range(1980, 2025):
        players = pd.read_csv(f"processed_data/{year}_player_stats.csv")
        players = players.drop(['Unnamed: 0'], axis=1)
        
        # Filter Players based on games played
        condition = (players['G'] >= 20) | (players['MP'] >= 20) | (players['PTS'] >= 6)
        players = players[condition]
        players.fillna(0, inplace=True)
        
        # Split numeric and non-numeric before z-score
        numeric = players.select_dtypes(include=[np.number])
        z_scores = numeric.apply(zscore, nan_policy='raise')
        non_numeric = players.select_dtypes(exclude=[np.number])
        zscored = pd.concat([non_numeric, z_scores], axis=1)
        
        # The YR column needs to be offset by 1 year to match with the All Star data from a seperate csv
        zscored['YR'] = year - 1
        
        all_players = pd.concat([zscored, all_players], axis=0)
        
    # Reset index
    all_players = all_players.reset_index()
    all_players = all_players.drop('index', axis=1)
    
    # Apply function
    all_players['PLAYER'] = all_players['PLAYER'].apply(lambda x: remove_accents(x))
    
    return all_players

def process_all_star_data():
    all_star = pd.read_csv("raw_data/final_data.csv")

    all_star['PLAYER'] = all_star.iloc[:, 0] + ' ' + all_star.iloc[:, 1]
    all_star.drop(columns=[all_star.columns[0], all_star.columns[1]], inplace=True)
    
    all_star.rename(columns={'year': 'Year'}, inplace=True)
    all_star.to_csv('processed_data/all_star.csv', index=False)

    all_star = pd.read_csv("processed_data/all_star.csv")

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('ASCII')
    return only_ascii

def add_all_stars():
    all_star_1979_80 = {
    'PLAYER': [
        'Tiny Archibald', 'Larry Bird', 'Bill Cartwright', 'Dave Cowens', 'John Drew',
        'Julius Erving', 'George Gervin', 'Elvin Hayes', 'Eddie Johnson', 'Moses Malone',
        'Micheal Ray Richardson', 'Dan Roundfield',  # Eastern Conference
        'Kareem Abdul-Jabbar', 'Otis Birdsong', 'Adrian Dantley', 'Walter Davis', 'World B. Free',
        'Dennis Johnson', 'Marques Johnson', 'Magic Johnson', 'Jack Sikma', 'Kermit Washington',
        'Scott Wedman', 'Paul Westphal'  # Western Conference
    ],
    'Year': [1979] * 24 
    }

    all_star_1979_80_df = pd.DataFrame(all_star_1979_80)

    all_star_2023_24 = {
        'PLAYER': [
            'LeBron James', 'Nikola Jokic', 'Kevin Durant', 'Luka Doncic', 'Shai Gilgeous-Alexander',  # Western Starters
            'Devin Booker', 'Stephen Curry', 'Anthony Davis', 'Anthony Edwards', 'Paul George', 'Kawhi Leonard', 'Karl-Anthony Towns',  # Western Reserves
            'Giannis Antetokounmpo', 'Jayson Tatum', 'Joel Embiid', 'Tyrese Haliburton', 'Damian Lillard',  # Eastern Starters
            'Bam Adebayo', 'Paolo Banchero', 'Jaylen Brown', 'Jalen Brunson', 'Tyrese Maxey', 'Donovan Mitchell', 'Julius Randle', 'Trae Young', 'Scottie Barnes'  # Eastern Reserves
        ],
        'Year': [2023] * 26  # All players are for the year 2024
    }

    all_star_2023_24_df = pd.DataFrame(all_star_2023_24)

    return pd.concat([all_star_1979_80_df, all_star_2023_24_df])

def main():
    # Change when you need to recombine all the data files
    combined = True
    if not combined:
        combine_rosters()

    all_players = combine_seasons()

    process_all_star_data()
    
    all_star = pd.read_csv('processed_data/all_star.csv')
    # We only need the player and year columns
    all_star = all_star[['PLAYER', 'Year']]
    
    added_all_stars = add_all_stars()

    all_star = pd.concat([all_star, added_all_stars])
    all_star = all_star.sort_values(by='Year', ascending=True)
    
    # Merge all_players with all_stars
    all_players['YR'] = all_players['YR'].astype(int)
    all_star['Year'] = all_star['Year'].astype(int)

    all_players = all_players.merge(all_star, how='left', left_on=['PLAYER', 'YR'], right_on=['PLAYER', 'Year'])

    # lambda function to set 0 & 1 for labeling
    all_players['ALLSTAR'] = all_players['Year'].apply(lambda x: 1 if pd.notnull(x) else 0)
    all_players.drop(columns=['Year'], inplace=True)

    all_players['YR'] = all_players['YR'] - 1979
    
    all_players.to_csv("processed_data/player_data.csv")
    
    print("Data saved to processed_data/player_data.csv")
    
if __name__ == '__main__':
    main()
    
