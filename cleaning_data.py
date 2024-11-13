import pandas as pd
import numpy as np
from datetime import datetime, date
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


# %%
# Calculates current age of fight at present.
def calculate_age(born):
    today = datetime.now()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def age_at_fight(born, date):
    return (date - born).astype('<m8[Y]')


def remove_other_side(col):
    return col[:-11] if col.endswith('_other_side') else col


def find_other_fighter(row):
    # Split the matchup into two competitors
    fighter = row['BOUT'].split(' vs. ')
    # Return the competitor that is not equal to the given competitor
    return fighter[1] if row['FIGHTER'] == fighter[0] else fighter[0]


# Function to determine the outcome for each competitor
def find_outcomes(row):
    competitors = row['BOUT'].split(' vs. ')
    outcome = row['OUTCOME']

    # Determine outcomes based on the matchup and outcome
    if outcome == 'W/L':
        competitor_outcome = 'W' if row['FIGHTER'] == competitors[0] else 'L'
        competitor_two_outcome = 'L' if row['FIGHTER'] == competitors[0] else 'W'
    else:  # outcome == 'L/W'
        competitor_outcome = 'L' if row['FIGHTER'] == competitors[0] else 'W'
        competitor_two_outcome = 'W' if row['FIGHTER'] == competitors[0] else 'L'

    return pd.Series([competitor_outcome, competitor_two_outcome])

    # Apply the function to create the new columns


def current_stats(dataframe):
    # Drop specified columns
    dataframe = dataframe.drop(
        columns=['EVENT', 'BOUT', 'OUTCOME', 'WEIGHT_opponent', 'REACH_opponent', 'STANCE_opponent',
                 'METHOD', 'ROUND', 'TIME FORMAT', 'FIGHTER2', 'fighter_outcome',
                 'fighter_outcome_2', 'LOCATION'])
    # Drop rows where 'DATE' is NaN
    dataframe = dataframe.dropna(subset=['DATE'])

    # Group by 'FIGHTER' and find the index of the most recent 'DATE'
    idx = dataframe.groupby('FIGHTER')['DATE'].idxmax()
    idx = idx.dropna()  # Drop any NaN values from the index

    # Select the rows with the most recent stats for each fighter
    fighter_current = dataframe.loc[idx]
    return fighter_current


def compare_fighters(fighter1, fighter2):
    fighter_a = new_stats[1].loc[new_stats[1]['FIGHTER'] == fighter1]
    fighter_b = new_stats[1].loc[new_stats[1]['FIGHTER'] == fighter2]
    comparative_stats = pd.concat([fighter_a, fighter_b])
    return comparative_stats


# %%
def fight_stats(stat_path, event_path, outcome_path, fighter_path):
    data = pd.read_csv(stat_path)
    print(f'{len(data) - len(data.dropna())} columns were dropped')

    # Drop the null rows
    data = data.dropna()
    # Drop the percentage columns, we can calculate these later. also drop round, we have use this in another dataframe
    data = data.drop(columns=['SIG.STR. %', 'TD %', 'ROUND'])
    to_split = ['SIG.STR.', 'TOTAL STR.', 'TD', 'HEAD', 'BODY', 'LEG', 'DISTANCE', 'CLINCH', 'GROUND']
    to_split_rename = ['sig_strike', 'total_strike', 'take_down', 'head_strike', 'body_strike', 'leg_strike',
                       'distance_strike', 'clinch_strike', 'ground_strike']

    # split and rename columns with 'landed of attempted' form
    for i in range(len(to_split)):
        data[[f'{to_split_rename[i]}_landed', f'{to_split_rename[i]}_attempted']] = data.pop(to_split[i]).str.split(
            ' of ', expand=True)

    # replace the blank ctrl time with 0:0 and split for minutes and second
    data['CTRL'] = data['CTRL'].replace('--', '0:0')
    data[['min_ctrl', 's_ctrl']] = data.pop('CTRL').str.split(':', expand=True)

    # convert str splits into integers
    data = data.astype('int', errors='ignore')

    # find total ctrl time in seconds
    data['total_ctrl_s'] = (data['min_ctrl'] * 60) + data['s_ctrl']
    data = data.drop(columns=['min_ctrl', 's_ctrl'])
    # sum the stats by fighter and bout so that the rounds are gone and we just have stats per fight
    data = data.groupby(['EVENT', 'BOUT', 'FIGHTER']).sum().reset_index()
    data = data.reset_index()

    # wrangle fight outcome data
    outcome = pd.read_csv(outcome_path)
    outcome['TIME'] = outcome['TIME'].astype(str)
    outcome[['DURATION_MIN', 'DURATION_SEC']] = outcome['TIME'].str.split(':', expand=True)
    outcome['DURATION_MIN'] = outcome['DURATION_MIN'].astype(float)
    outcome['DURATION_SEC'] = outcome['DURATION_SEC'].astype(float)
    outcome['TIME'] = (outcome['DURATION_MIN'] * 60) + outcome['DURATION_SEC']
    outcome['TIME'] = ((outcome['ROUND'].astype(float) - 1) * 5 * 60) + outcome['TIME']
    outcome = outcome.drop(columns=['DETAILS', 'URL', 'REFEREE', 'DURATION_MIN', 'DURATION_SEC'])
    outcome = outcome.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    outcome['BOUT'] = outcome['BOUT'].str.replace(r'\s+', ' ', regex=True)
    data = pd.merge(data, outcome, on=['EVENT', 'BOUT'], how='left')
    data = data.drop_duplicates(subset=['EVENT', 'BOUT', 'FIGHTER'], keep='first')
    data['FIGHTER2'] = data.apply(find_other_fighter, axis=1)
    data[['fighter_outcome', 'fighter_outcome_2']] = data.apply(find_outcomes, axis=1)

    # make opponent dataframe for defensive stats
    o_data = data[
        ['EVENT', 'BOUT', 'FIGHTER', 'FIGHTER2', 'KD', 'SUB.ATT', 'REV.', 'sig_strike_landed', 'sig_strike_attempted',
         'total_strike_landed', 'total_strike_attempted', 'take_down_landed', 'take_down_attempted',
         'head_strike_landed', 'head_strike_attempted', 'body_strike_landed', 'body_strike_attempted',
         'leg_strike_landed', 'leg_strike_attempted', 'distance_strike_landed', 'distance_strike_attempted',
         'clinch_strike_landed', 'clinch_strike_attempted', 'ground_strike_landed', 'ground_strike_attempted',
         'total_ctrl_s']]

    o_data = o_data.rename(columns={'FIGHTER': 'FIGHTER2', 'FIGHTER2': 'FIGHTER'})

    data = pd.merge(data, o_data, left_on=['EVENT', 'BOUT', 'FIGHTER', 'FIGHTER2'],
                    right_on=['EVENT', 'BOUT', 'FIGHTER', 'FIGHTER2'], suffixes=('_fighter', '_by_opponent'))

    # wrangle events data:
    events = pd.read_csv(event_path)
    events['DATE'] = pd.to_datetime(events['DATE'], format='%B %d, %Y')
    events[['CITY', 'STATE', 'COUNTRY']] = events['LOCATION'].str.split(',', expand=True)
    events = events.drop(columns=['STATE', 'COUNTRY', 'CITY', 'URL'])
    events = events.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    events = events[(events['DATE'] >= '2000-01-01')]
    data = pd.merge(data, events, on=['EVENT'], how='left')

    # Calculate the cumulative sum of stats for each person
    # (we don't want round to be summed, it is informative the way it is)
    data['ROUND'] = data['ROUND'].astype('str')
    map_wl = {'W': 1,
              'L': 0}
    data['loss_count'] = data['fighter_outcome_2'].map(map_wl)
    data['win_count'] = data['fighter_outcome'].map(map_wl)

    data = data.sort_values(['FIGHTER', 'DATE'], ascending=[True, True])
    accumulated_columns = []
    columns_to_accumulate = []
    for column in list(data.columns):
        if data[column].dtype == 'int':
            columns_to_accumulate.append(column)
            accumulated_columns.append(f'{column}_total')
        elif data[column].dtype == 'float':
            columns_to_accumulate.append(column)
            accumulated_columns.append(f'{column}_total')
        else:
            continue

    rolling_last_two = [f'{thing}_last_two' for thing in columns_to_accumulate]
    rolling_last_three = [f'{thing}_last_three' for thing in columns_to_accumulate]
    data[rolling_last_two] = data.groupby('FIGHTER')[columns_to_accumulate].rolling(window=2).sum().reset_index(level=0,
                                                                                                                drop=True)
    data[rolling_last_three] = data.groupby('FIGHTER')[columns_to_accumulate].rolling(window=3).sum().reset_index(
        level=0, drop=True)
    data[accumulated_columns] = data.groupby('FIGHTER')[columns_to_accumulate].cumsum()

    # wrangling individual fighter data
    fighters = pd.read_csv(fighter_path)
    fighters['STANCE'] = fighters['STANCE'].fillna('Orthodox')
    fighters['STANCE'] = fighters['STANCE'].replace('--', 'Orthodox')
    fighters = fighters[~fighters.apply(lambda x: x.str.contains("--", na=False)).any(axis=1)]
    fighters['DOB'] = pd.to_datetime(fighters['DOB'], format="%b %d, %Y")
    fighters[['height_feet', 'height_inches']] = fighters.pop('HEIGHT').str.split("' ", expand=True)
    fighters['height_inches'] = fighters['height_inches'].str.replace('"', '')
    fighters['HEIGHT'] = (fighters['height_feet'].astype('int') * 12) + fighters['height_inches'].astype('int')
    fighters['WEIGHT'] = fighters['WEIGHT'].str.replace(' lbs.', '')
    fighters['REACH'] = fighters['REACH'].str.replace('"', '')
    fighters = fighters.drop(columns=['height_inches', 'height_feet', 'URL'])
    fighters = fighters.drop_duplicates(subset=['FIGHTER'])
    opponents = fighters.rename(columns={'FIGHTER': 'FIGHTER2'})
    data = pd.merge(data, fighters, on='FIGHTER', how='left')
    data = pd.merge(data, opponents, on='FIGHTER2', how='left', suffixes=('_fighter', '_opponent'))
    data['age_fighter'] = ((((data['DATE'] - data['DOB_fighter']).astype('<m8[s]')).astype(int)) / 31556952).astype(int)

    # to gather stats for proportions
    landed = []
    attempted = []
    for thing in to_split_rename:
        for col in list(data.columns):
            if f'{thing}_landed' in col:
                landed.append(col)
            elif f'{thing}_attempted' in col:
                attempted.append(col)
            else:
                continue

    for thing in to_split_rename:
        for i in range(len(attempted)):
            if thing in attempted[i]:
                if 'opponent' in attempted[i]:
                    if '_last_three' in attempted[i]:
                        data[f'{thing}_opponent_last_three_proportion'] = data[landed[i]] / data[attempted[i]]
                    elif '_last_two' in attempted[i]:
                        data[f'{thing}_opponent_last_two_proportion'] = data[landed[i]] / data[attempted[i]]
                    elif '_total' in attempted[i]:
                        data[f'{thing}_opponent_total_proportion'] = data[landed[i]] / data[attempted[i]]
                    else:
                        data[f'{thing}_opponent_last_proportion'] = data[landed[i]] / data[attempted[i]]
                else:
                    if '_last_three' in attempted[i]:
                        data[f'{thing}_last_three_proportion'] = data[landed[i]] / data[attempted[i]]
                    elif '_last_two' in attempted[i]:
                        data[f'{thing}_last_two_proportion'] = data[landed[i]] / data[attempted[i]]
                    elif '_total' in attempted[i]:
                        data[f'{thing}_total_proportion'] = data[landed[i]] / data[attempted[i]]
                    else:
                        data[f'{thing}_last_proportion'] = data[landed[i]] / data[attempted[i]]
            else:
                continue

    stats1 = []
    for thing in list(data.columns):
        if data[thing].dtype == 'object':
            continue
        if data[thing].dtype == 'datetime64[ns]':
            continue
        else:
            if '_fighter' in thing:
                print(f"{thing} is dtype {data[thing].dtype}")
                stats1.append(thing)
            if "_opponent" in thing:
                print(f"{thing} is dtype {data[thing].dtype}")
                stats1.append(thing)
            else:
                continue
    # stats that will get additional rate, eg strike/min

    for thing in stats1:
        if '_total' in thing:
            data[f'{thing}_per_min'] = data[thing] / (data['TIME_total'] / 60)
        elif '_last_two' in thing:
            data[f'{thing}_per_min'] = data[thing] / (data['TIME_last_two'] / 60)
        elif '_last_three' in thing:
            data[f'{thing}_per_min'] = data[thing] / (data['TIME_last_three'] / 60)
        else:
            data[f'{thing}_per_min'] = data[thing] / (data['TIME'] / 60)

    for cols in data.columns:
        if 'index' in cols:
            data.pop(cols)
        else:
            continue

    # categorical data to numeric
    categorical_to_numeric = ['FIGHTER', 'FIGHTER2', 'STANCE_fighter', 'STANCE_opponent', 'TIME FORMAT']
    for thing in categorical_to_numeric:
        data[f'{thing.lower()}_codes'] = data[thing].astype('category').cat.codes

    # get current stats for predictions
    current = current_stats(data)

    data = data.drop(columns=[])
    return data, current
