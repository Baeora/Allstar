# import libraries 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import seaborn as sns 
import numpy as np
import mplcyberpunk

from core import *

# Read XL Data
data = pd.read_excel('private\\test_data.xlsx')
country_codes=list(data['Country code'].unique())
months = pd.DataFrame({'cohort_month':add_cohorts(data)['month'].sort_values().unique()})

##############
### Tables ###
##############

# Cohort Retention Analysis
def get_retention_table(df, pct=True):

    df = add_cohorts(df=df)

    # Create a new dataframe containing a list of how many unique users are in each cohort for each month
    # Each cohort is defined by how many months a user is active after said month
    cohort_data = df.groupby(['cohort_month','cohort_index'])['User ID'].apply(pd.Series.nunique).reset_index()

    cohort_table = cohort_data.pivot(index='cohort_month', columns=['cohort_index'],values='User ID')
    cohort_table = months.set_index('cohort_month').join(cohort_table)
    for i in range(len(cohort_table)-1):
        cohort_table.iloc[i] = cohort_table.iloc[i].fillna(0)
    for i in range(len(cohort_table.columns)):
        if i > 0:
            cohort_table.iloc[len(cohort_table)-i:len(cohort_table),i] = np.NaN
    
    cohort_sizes = cohort_table.iloc[:,0]

    if pct is False:
        return cohort_table

    # Make a new table where we are dividing the values of each row by the size of the cohort of that row
    # This gives us a percentage of the cohort that is still active over time
    retention_table = cohort_table.divide(cohort_sizes,axis=0)
    retention_table = months.set_index('cohort_month').join(retention_table)

    # Replacing first column with raw cohort size instead of it always being 100% (More useful)
    retention_table[1] = cohort_sizes

    return retention_table

# Monthly Churn Rate
def get_churn_table(df):

    rt_table = get_retention_table(df, pct=False)
    
    cols_reversed = list(range(len(rt_table.columns)))
    cols_reversed.reverse()
    for i in cols_reversed:
        if i > 0:
            rt_table.iloc[:,i] -= rt_table.iloc[:,i-1]

    df = rt_table.iloc[:,1:].fillna(0)

    c = df.to_numpy()
    vals = [np.sum(np.diag(np.flipud(c), k=i)) for i in range(-len(df) + 1, len(df), 1)]

    churned_users = vals[0:len(rt_table)]
    churn = []

    sub_change = pd.DataFrame(pd.Series(churned_users[:-1]) + pd.Series(rt_table[1][1:].reset_index()[1]))
    sub_change['total_change'] = 0
    for i in range(len(sub_change)):
        if i > 0:
            sub_change.loc[i, 'total_change'] = sub_change[0][i] + sub_change['total_change'][i-1]
        else:
            sub_change.loc[i, 'total_change'] = sub_change[0][i]

    blank_row = pd.DataFrame({0:[0], 'total_change':[0]})
    sub_change = pd.concat([blank_row, sub_change]).reset_index()

    for i in range(len(rt_table)):
        date = rt_table.index[i]
        if (rt_table[1][0] + sub_change['total_change'][i]) > 0:
            churn_rate = (churned_users[i] / (rt_table[1][0] + sub_change['total_change'][i]))
        else:
            churn_rate = np.NaN
        churn.append((date, churn_rate, (rt_table[1][0] + sub_change['total_change'][i])))

    churn_table = pd.DataFrame(churn, columns=['cohort_month', 'churn_rate', 'current_subs']).set_index('cohort_month')
    churn_table.insert(1,'new_subs', rt_table[1])

    churned_users.insert(0,0)
    churn_table.insert(1,'lost_subs', churned_users[0:-1])

    return churn_table

# Net Dollar Retention
def get_ndr_table(df):

    cohort_df = add_cohorts(df)

    revenue = cohort_df.groupby(['cohort_month', 'cohort_index']).agg({'Simple checkout purchase amount':'sum'}).reset_index()
    revenue_table = revenue.pivot(index='cohort_month', columns=['cohort_index'],values='Simple checkout purchase amount')

    base_revenue = revenue_table.iloc[:,0]

    ndr_table = revenue_table.divide(base_revenue,axis=0)
    ndr_table[1] = base_revenue 

    ndr_table = months.set_index('cohort_month').join(ndr_table)
    ndr_table.index = ndr_table.index.strftime('%B %Y')

    return round(ndr_table, 2)

# Customer Lifetime Revenue
def get_clr_table(df):

    cohort_df = add_cohorts(df)

    cohort_size = cohort_df.groupby(['cohort_month', 'cohort_index']).agg({'User ID':'count'}).reset_index()
    cohort_table = cohort_size.pivot(index='cohort_month', columns=['cohort_index'],values='User ID')

    revenue = cohort_df.groupby(['cohort_month', 'cohort_index']).agg({'Simple checkout purchase amount':'sum'}).reset_index()
    revenue_table = revenue.pivot(index='cohort_month', columns=['cohort_index'],values='Simple checkout purchase amount')

    for i in range(len(revenue_table)-1):
        revenue_table.iloc[i] = revenue_table.iloc[i].fillna(0)
    for i in range(len(revenue_table.columns)):
        if i > 0:
            revenue_table.iloc[len(revenue_table)-i:len(revenue_table),i] = np.NaN

    for i in range(len(revenue_table.columns)):
        if i > 0:
            revenue_table.iloc[:,i] += revenue_table.iloc[:,i-1]

    clr_table = revenue_table.divide(cohort_table,axis=1)
    clr_table.iloc[:,0] = cohort_table.iloc[:,0]
    clr_table = clr_table.rename(columns={1:'cohort_size'})
    clr_table = months.set_index('cohort_month').join(clr_table)
    clr_table.index = clr_table.index.strftime('%B %Y')

    return round(clr_table, 2)

# Customer Lifetime Value
def get_clv_table(df):

    gross_margin = df['Total payout amount, %'].mean()/100

    clv_table = get_clr_table(df)
    clv_table.iloc[:,1:len(clv_table)] = round(clv_table.iloc[:,1:len(clv_table)] * gross_margin, 2)

    return clv_table

def get_plan_tier_breakdown_table(df):
    
    pro = get_churn_table(df[df['Subscription plan name'].isin(sub_dict['pro_monthly'])].copy())
    pro_plus = get_churn_table(df[df['Subscription plan name'].isin(sub_dict['pro_plus_monthly'])].copy())
    platinum = get_churn_table(df[df['Subscription plan name'].isin(sub_dict['platinum_monthly'])].copy())

    df = pd.merge(pd.DataFrame(pro['new_subs']), pd.DataFrame(pro_plus['new_subs']), how='inner', left_index=True, right_index=True)
    df = pd.merge(df, pd.DataFrame(platinum['new_subs']), how='inner', left_index=True, right_index=True)
    df.columns = ['pro_monthly', 'pro_plus_monthly', 'platinum_monthly']

    df['pro_pct'] = df['pro_monthly'] / (df['pro_monthly'] + df['pro_plus_monthly'] + df['platinum_monthly']) 
    df['pro_plus_pct'] = df['pro_plus_monthly'] / (df['pro_monthly'] + df['pro_plus_monthly'] + df['platinum_monthly']) 
    df['platinum_pct'] = df['platinum_monthly'] / (df['pro_monthly'] + df['pro_plus_monthly'] + df['platinum_monthly']) 

    return df

################
### Heatmaps ###
################

# Cohort Retention Analysis
def get_retention_heatmap(df, add_churn=True, savefig=False, path='', background='#0d1a28', facecolor='#0a1521'):

    retention_table = get_retention_table(df, pct=True)

    if add_churn:

        churn_table = pd.DataFrame(get_churn_table(df)['churn_rate'])
        churn_table = churn_table.join(retention_table)

        churn_table.iloc[:,0] = round(churn_table.iloc[:,0], 2)*100
        churn_table.iloc[:,0] = churn_table.iloc[:,0].fillna(0).astype(int)
        churn_table.iloc[:,0][-1] = np.NaN

        retention_table = churn_table

    retention_table.rename(columns={1:'cohort_size'}, inplace=True)
    retention_table.index = retention_table.index.strftime('%B %Y')
    retention_table = retention_table[retention_table['cohort_size'] > 0]

    # Grabbing dataframe x and y axis lengths
    x = len(retention_table.columns)
    y = len(retention_table)
    
    pct_mask = np.zeros((y, x), dtype=bool)
    pct_mask[:,0] = True

    # Mask for the values I wish to be formatted as 'g'
    count_mask = np.zeros((y, x), dtype=bool)
    count_mask[:,1:] = True

    if add_churn:

        pct_mask = np.zeros((y, x), dtype=bool)
        pct_mask[:,0:2] = True

        # Mask for the values I wish to be formatted as 'g'
        churn_mask = np.zeros((y, x), dtype=bool)
        churn_mask[:,1:] = True

        # Mask for the values I wish to be formatted as 'g'
        count_mask = np.zeros((y, x), dtype=bool)
        count_mask[:,2:] = True
        count_mask[:,0] = True

    # Plotting two heatmaps stacked on top of one another
    # One uses .0% formatting with the above pct_mask, other 'g' formatting with the count_mask
    # Count heatmap uses a different color than the % heatmap for visibility, and has cbar removed to reduce clutter
    plt.figure(figsize=(x+3,round(y*0.7))).set_facecolor(background)

    df_formatted = retention_table.map(
        lambda val: f'{val}%')

    sns.heatmap(retention_table, mask=pct_mask, annot=True, vmin=0, vmax=1, cbar=False, fmt='.0%').set_facecolor(facecolor)
    sns.heatmap(retention_table, mask=count_mask, annot=True, cbar=False, cmap='Blues', fmt='g').set_facecolor(facecolor)
    if add_churn:
        sns.heatmap(retention_table, mask=churn_mask, annot=df_formatted, cbar=False, cmap='Greens', fmt='').set_facecolor(facecolor)

    # Setting some simple plt params
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(False)
    plt.tick_params(axis='both', colors='w')

    if savefig:
        plt.savefig(f'{path}', bbox_inches='tight')

# Net Dollar Retention
def get_ndr_heatmap(df, savefig=False, path='', background='#0d1a28', facecolor='#0a1521'):

    ndr_table = get_ndr_table(df)

    # Grabbing dataframe x and y axis lengths
    x = len(ndr_table.columns)
    y = len(ndr_table)

    # Mask for the values I wish to be formatted as '.0%'
    pct_mask = np.zeros((y, x), dtype=bool)
    pct_mask[:,0] = True

    # Mask for the values I wish to be formatted as 'g'
    count_mask = np.zeros((y, x), dtype=bool)
    count_mask[:,1:] = True

    # Plotting two heatmaps stacked on top of one another
    # One uses .0% formatting with the above pct_mask, other 'g' formatting with the count_mask
    # Count heatmap uses a different color than the % heatmap for visibility, and has cbar removed to reduce clutter
    plt.figure(figsize=(x+3,round(y*0.7))).set_facecolor(background)

    df_formatted = ndr_table.map(
        lambda val: f'${val}')

    sns.heatmap(ndr_table, mask=pct_mask, annot=True, cbar=False, fmt='.0%').set_facecolor(facecolor)
    sns.heatmap(ndr_table, mask=count_mask, annot=df_formatted, cbar=False, cmap='Blues', fmt='').set_facecolor(facecolor)

    # Setting some simple plt params
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(False)
    plt.tick_params(axis='both', colors='w')

    if savefig:
        plt.savefig(f'{path}', bbox_inches='tight')

# Customer Lifetime Revenue
def get_clr_heatmap(df, savefig=False, path='', background='#0d1a28', facecolor='#0a1521'):

    clr_table = get_clr_table(df)

    # Grabbing dataframe x and y axis lengths
    x = len(clr_table.columns)
    y = len(clr_table)

    # Mask for the values I wish to be formatted as '.0%'
    pct_mask = np.zeros((y, x), dtype=bool)
    pct_mask[:,0] = True

    # Mask for the values I wish to be formatted as 'g'
    count_mask = np.zeros((y, x), dtype=bool)
    count_mask[:,1:] = True

    # Plotting two heatmaps stacked on top of one another
    # One uses .0% formatting with the above pct_mask, other 'g' formatting with the count_mask
    # Count heatmap uses a different color than the % heatmap for visibility, and has cbar removed to reduce clutter
    plt.figure(figsize=(x+3,round(y*0.7))).set_facecolor(background)

    df_formatted = clr_table.map(
        lambda val: f'${val}')

    sns.heatmap(clr_table, mask=pct_mask, annot=df_formatted, cbar=False, fmt='').set_facecolor(facecolor)
    sns.heatmap(clr_table, mask=count_mask, annot=True, cbar=False, cmap='Blues', fmt='').set_facecolor(facecolor)

    # Setting some simple plt params
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(False)
    plt.tick_params(axis='both', colors='w')

    if savefig:
        plt.savefig(f'{path}', bbox_inches='tight')

# Customer Lifetime Value
def get_clv_heatmap(df, savefig=False, path='', background='#0d1a28', facecolor='#0a1521'):

    clv_table = get_clv_table(df)

    # Grabbing dataframe x and y axis lengths
    x = len(clv_table.columns)
    y = len(clv_table)

    # Mask for the values I wish to be formatted as '.0%'
    pct_mask = np.zeros((y, x), dtype=bool)
    pct_mask[:,0] = True

    # Mask for the values I wish to be formatted as 'g'
    count_mask = np.zeros((y, x), dtype=bool)
    count_mask[:,1:] = True

    # Plotting two heatmaps stacked on top of one another
    # One uses .0% formatting with the above pct_mask, other 'g' formatting with the count_mask
    # Count heatmap uses a different color than the % heatmap for visibility, and has cbar removed to reduce clutter
    plt.figure(figsize=(x+3,round(y*0.7))).set_facecolor(background)

    df_formatted = clv_table.map(
        lambda val: f'${val}')

    sns.heatmap(clv_table, mask=pct_mask, annot=df_formatted, cbar=False, fmt='').set_facecolor(facecolor)
    sns.heatmap(clv_table, mask=count_mask, annot=True, cbar=False, cmap='Blues', fmt='').set_facecolor(facecolor)

    # Setting some simple plt params
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(False)
    plt.tick_params(axis='both', colors='w')

    if savefig:
        plt.savefig(f'{path}', bbox_inches='tight')

############
### Misc ###
############

def get_monthly_sub_count_chart(df, sub_type=None, savefig=False, path=''):

    if sub_type is not None:
        df = df[df['Subscription plan name'].isin(sub_dict[sub_type])].copy()

        df = get_churn_table(df)
        df = df[df['current_subs'] > 0]
        lost_subs = df["lost_subs"]*-1/2
        new_subs = df['new_subs']/2
        current_subs = df['current_subs']
        x = list(df.index)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use("cyberpunk")
        plt.grid(c='#0d1a28')
        plt.setp(ax.spines.values(), visible=False)
        plt.tick_params(axis='both', colors='w')
        fig.set_facecolor('#0d1a28')
        ax.set_facecolor('#0a1521')
        ax.set_axisbelow(True)

        myFmt = DateFormatter('%m-%Y')
        ax.xaxis.set_major_formatter(myFmt)

        ax.plot(x, current_subs, marker='o', color='#08eaf2')
        new_sub_bar = ax.bar(x, new_subs, bottom=lost_subs, color='#08eaf2', width = 20)
        lost_sub_bar = ax.bar(x, lost_subs, color='#f04f75', width = 20)
        
        mplcyberpunk.add_bar_gradient(bars=new_sub_bar)
        mplcyberpunk.add_bar_gradient(bars=lost_sub_bar)

        mplcyberpunk.make_lines_glow(ax, n_glow_lines=3)
        mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
        
        plt.show()

        if savefig:
            plt.savefig(f'{path}', bbox_inches='tight')
    else:
        pro = df[df['Subscription plan name'].isin(sub_dict['pro_monthly'])].copy()
        pro_plus = df[df['Subscription plan name'].isin(sub_dict['pro_plus_monthly'])].copy()
        platinum = df[df['Subscription plan name'].isin(sub_dict['platinum_monthly'])].copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use("cyberpunk")
        plt.grid(c='#0d1a28')
        plt.setp(ax.spines.values(), visible=False)
        plt.tick_params(axis='both', colors='w')
        fig.set_facecolor('#0d1a28')
        ax.set_facecolor('#0a1521')
        ax.set_axisbelow(True)

        myFmt = DateFormatter('%m-%Y')
        ax.xaxis.set_major_formatter(myFmt)

        for df, color in [(pro, '#08eaf2'), (pro_plus, '#f04f75'), (platinum, '#ffd203')]:

            df = get_churn_table(df)

            df = df[df['current_subs'] > 0]

            current_subs = df['current_subs']
            x = list(df.index)

            ax.plot(x, current_subs, marker='o', color=color)

            
        mplcyberpunk.make_lines_glow(ax, n_glow_lines=3)
        mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
        plt.show()

def get_plan_tier_pie_chart(df, past_months=3):
    
    plans = get_plan_tier_breakdown_table(df)

    plan_pcts = plans[-past_months:len(plans)].mean()

    plan_list = [plan_pcts['pro_pct'], plan_pcts['pro_plus_pct'], plan_pcts['platinum_pct']]

    #colors
    colors = ['#08eaf2','#f04f75','#ffd203']
    labels = ['Pro', 'Pro Monthly', 'Platinum']

    fig, ax = plt.subplots()
    fig.set_facecolor('#0d1a28')

    explode = (0.05,0.05,0.05)
    patches, texts, autotexts = ax.pie(plan_list, colors=colors, autopct='%1.1f%%', labels=labels, startangle=90, pctdistance=0.75, explode=explode)
    for autotext in autotexts:
        autotext.set_color('black')
    centre_circle = plt.Circle((0,0),0.60,fc='#0d1a28')

    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle

    ax.axis('equal')  
    plt.tight_layout()
    plt.show()

def get_regional_chart(df):
    df['Country code'].value_counts()

    country_count  = df['Country code'].value_counts()
    country_count = country_count[:15,]
    background='#0d1a28'
    facecolor='#0a1521'
    plt.figure(figsize=(10,5)).set_facecolor(background)

    # Setting some simple plt params

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("cyberpunk")
    plt.grid(c='#0d1a28')
    plt.setp(ax.spines.values(), visible=False)
    plt.tick_params(axis='both', colors='w')
    
    fig.set_facecolor('#0d1a28')
    ax.set_facecolor('#0a1521')
    ax.set_axisbelow(True)

    sns.barplot(x=country_count.values, y=country_count.index, alpha=1, orient='h', palette="crest").set_facecolor(facecolor)
    for i in ax.containers:
        ax.bar_label(i, padding=5)
    plt.ylabel(None)
    plt.xlabel(None)
 
if __name__ == '__main__':

    df = clean_data(data)

    for sub_type in list(sub_dict.keys()):
        cohort_df = df[df['Subscription plan name'].isin(sub_dict[sub_type])].copy()
        if 'monthly' in sub_type:
            add_churn=True
        else:
            add_churn=False
        get_retention_heatmap(cohort_df, add_churn=add_churn, savefig=True, path=f'plots\\cohort_analyses\\retention\\sub_type\\{sub_type}')

    for country in country_codes:
        cohort_df = df[df['Country code'].isin([country]) & 
                    (
                        (df['Subscription plan name'].isin(sub_dict['pro_monthly'])) | 
                        (df['Subscription plan name'].isin(sub_dict['pro_plus_monthly'])) |
                        (df['Subscription plan name'].isin(sub_dict['platinum_monthly']))
                    )].copy()
        if len(cohort_df) > 1:              
            get_retention_heatmap(cohort_df, add_churn=True, savefig=True, path=f'plots\\cohort_analyses\\retention\\country\\{country}')
