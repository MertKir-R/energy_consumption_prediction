import numpy as np
import pandas as pd

def linear_interpolation(data, col): 
    valcnt = data['country'].value_counts() 
    nullcnt = data[data[col].isna()]['country'].value_counts()
    merge_null = pd.DataFrame({
        'total_count': valcnt,    
        'null_count': nullcnt     
    }).fillna(0)
    
    all_null = merge_null[merge_null['total_count'] == merge_null['null_count']].index 
    # all_null = (data.groupby('country')[col].apply(lambda s: s.isna().all()))
    
    first_occur = data[~data[col].isna()].groupby('country')['year'].min().rename('min_year') 
    data_merge = data.merge(first_occur, on = "country", how = "left") 
    data_merge = data_merge.sort_values(['country','year']) 
    
    zero_index = data_merge[data_merge['min_year'] > data_merge['year']].index 
    data_merge.loc[zero_index, col] = 0 

    all_null_index = data_merge[data_merge['country'].isin(all_null)].index
    data_merge.loc[all_null_index, col] = 0 
    
    interp = data_merge.groupby('country')[col].transform(lambda s: s.interpolate(limit_area='inside'))
    data_merge[col] = data_merge[col].fillna(interp)

    data_merge = data_merge.drop(columns='min_year')
        
    return data_merge


def fill_extrapolation(df, col):
    na_countries = df[df[col].isna()]['country'].unique()
    min_year = df[df[col].isna()]['year'].unique().min()-1

    country_list = df[(df['country'].isin(na_countries)) & (df['year'] == min_year) & (df[col] > 0)]['country'].tolist()
    zero_country_list = df[(df['country'].isin(na_countries)) & (df['year'] == min_year) & (df[col] == 0)]['country'].tolist()
    extrapolate_df = df.copy()
    extrapolate_df = extrapolate_df.sort_values(['country', 'year'])

    zero_country_index = extrapolate_df[extrapolate_df['country'].isin(zero_country_list)].index
    extrapolate_df.loc[zero_country_index, col] = 0

    for country in country_list:
        temp = extrapolate_df[(extrapolate_df['country'] == country) & (extrapolate_df['year'] <= min_year)].sort_values('year')
        recent = temp.tail(7) # use last 7 years
        # Fit a simple linear model y = a*year + b
        a, b = np.polyfit(recent['year'].to_numpy(dtype=float), recent[col].to_numpy(dtype=float), 1)

        future = extrapolate_df[(extrapolate_df['country'] == country) &
                                (extrapolate_df['year'] > min_year) &
                                (extrapolate_df[col].isna())]
        
        preds = a * future['year'].to_numpy(dtype=float) + b # predicts all the missing years with the linear trend from a and b
        extrapolate_df.loc[future.index, col] = np.clip(preds, 0, None) # If any predicted value is negative, replace it with 0

    return extrapolate_df
 
