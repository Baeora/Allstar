import pandas as pd
from datetime import datetime as dt

sub_dict = {
    # Monthly Subscribers
    'pro_monthly':['Allstar Pro - Monthly', 'Pro - Monthly'],
    'pro_plus_monthly':['Allstar Pro Plus - Monthly', 'Pro Plus - Monthly'],
    'platinum_monthly':['Allstar Creator Platinum - Monthly', 'Platinum - Monthly'],

    # Quarterly Subscribers
    'pro_quarterly':['Allstar Pro - Quarterly', 'Pro - Quarterly'],
    'pro_plus_quarterly':['Allstar Pro Plus - Quarterly', 'Pro Plus - Quarterly'],
    'platinum_quarterly':['Allstar Creator Platinum - Quarterly', 'Platinum - Quarterly'],

    # Biannual Subscribers
    'pro_biannually':['Allstar Pro - Biannually', 'Pro - Biannually'],
    'pro_plus_biannually':['Allstar Pro Plus - Biannually', 'Pro Plus - Biannually'],
    'platinum_biannually':['Allstar Creator Platinum - Biannually', 'Platinum - Biannually'],

    # Yearly Subscribers
    'pro_yearly':['Allstar Pro - Yearly', 'Pro - Yearly'],
    'pro_plus_yearly':['Allstar Pro Plus - Yearly', 'Pro Plus - Yearly'],
    'platinum_yearly':['Allstar Creator Platinum - Yearly', 'Platinum - Yearly']
}

# Acceptable Values
vals = []
vals_pro_monthly = [3.99, 3.39]
vals_pro_plus_monthly = [9.99, 8.49]
vals_platinum_monthly = [24.99, 21.25]

vals_pro_yearly = [47.99, 47.88, 23.94]
vals_pro_plus_yearly = [119.99, 119.88, 59.99]
vals_platinum_yearly = [299.99, 299.88, 149.94]

sub_zip = [(vals_pro_monthly, 'pro_monthly'),
           (vals_pro_plus_monthly, 'pro_plus_monthly'),
           (vals_platinum_monthly, 'platinum_monthly'),
           (vals_pro_yearly, 'pro_yearly'),
           (vals_pro_plus_yearly, 'pro_plus_yearly'),
           (vals_platinum_yearly, 'platinum_yearly'),]

web_mapper = {
    '/clips?external=':'clip_scroll',
    '/clips?game=':'clip_scroll',
    '/clips?filter=':'clip_scroll',
    '/clips?utm_source=':'clip_scroll',
    '/clips-gdn':'clip_scroll',
    '/clips-yt':'clip_scroll',
    '/clips-fb':'clip_scroll',
    './':'clip_scroll',
    'https://allstargg-web.pages.dev/clips':'clip_scroll',
    '/valorantclips':'clip_scroll',
    '/leagueoflegendsclips':'clip_scroll',
    '/dotaclips':'clip_scroll',

    '/u/':'user_scroll',
    '/i/':'user_scroll',
    '/id/':'user_scroll',
    '/users/':'user_scroll',
    '/clips/':'user_scroll',
    '/clips?fbclid=':'user_scroll',

    '/clip?':'clip_view',
    '/clip=':'clip_view',
    '/iframe?clip=':'clip_view',
    
    '/oauth/steam':'auth',
    '/oauth/faceit':'auth',
    '/oauth/discord':'auth',
    '/oauth/tiktok':'auth',
    '/oauth/google':'auth',
    
    '/authenticate/discord':'auth',
    '/authenticate/steam':'auth',
    '/authenticate/faceit':'auth',
    '/authenticate/tiktok':'auth',
    '/authenticate/google':'auth',

    
    '/iframe':'iframe',
    '/iframe?platform=':'iframe',
    '/iframe/competitions':'iframe',
    '/competition':'iframe',
    '/competitions':'iframe',
    'iframe?static=':'iframe',

    '/overwolf':'overwolf',

    '/connectedaccounts':'connected_accounts',
    '/profile?user=':'own_profile_view',
    '/profile':'own_profile_view',

    '/match-history': 'match-history',

    '/signup':'account_management',
    '/accountsettings':'account_management',
    '/register':'account_management',
    '/setup':'account_management',
    '/verify':'account_management',
    '/reset':'account_management',
    '/finishsignup?code=':'account_management',
    '/referred':'account_management',

    '/login':'login',
    '/logout':'logout',

    '/dashboard':'dashboard',
    '/notifications':'dashboard',
    '/subscriptions':'dashboard',
    '/myallstar':'dashboard',

    '/404':'404',
    '/help':'help',
    
    '/allstarbot':'discord_bot',
    '/discordinvite':'discord_bot',

    '/legal/':'site_exploration',
    '/culture':'site_exploration',
    '/howitworks':'site_exploration', 
    '/about':'site_exploration',
    '/profit':'site_exploration',
    '/Profit':'site_exploration',
    '/support':'site_exploration',
    '/whatsnew':'site_exploration',
    '/faq':'site_exploration',
    '/brand':'site_exploration',
    '/giveaways':'site_exploration',
    '/press':'site_exploration',
    '/careers':'site_exploration',
    '/email-signup':'site_exploration',
    '/mobileplaystore':'site_exploration',

    '/gamersclub':'gamers_club',
    '/upgrade':'upgrade',
    '/studio':'studio',
    '/feature/allstarstudio':'studio',
    '/remix':'remix',
    '/partnerprogram':'partner_program',
    '/superstar':'superstar',
    '/m/':'montage_scroll',
    '/montages':'montage_scroll',
    '/montage':'montage_scroll',
    '/montage?montage=':'montage_view',
    '/ymontages/create':'montage_create',
    '/claim':'partner_claim',

    
    '/?utm_campaign=':'unknown',
    '/openid.identity=':'unknown',
    '/.com':'unknown',
    '?ct=':'unknown',
    '/97965538015':'unknown',
    '/%7Bbase_name%7D':'unknown',
    '/sitemap.xml':'unknown',
    '/mobile-app-link':'unknown',
    '/group/':'unknown',
    '/bin/':'unknown',
    '/app-ads.txt':'unknown',
    '/feature/assets':'unknown',
    '?ajs_event=':'unknown',
    '?nitroads_debug=1':'unknown',
    '/60&openid':'unknown',
    '/?fbclid=':'unknown',
    '/.well-known':'unknown',
    '/hc/en-us':'unknown',
    '/apple-app-site-association':'unknown',
    '/interstitial':'unknown',
}

# Not segmenting by game for now, can do at a later point
desktop_mapper = {
    'CLIPS':'clip_scroll',
    'VIEW_CLIP':'clip_view',
    'Activity Feed':'clip_scroll',
    'VIEW_MOBILE_CLIP':'clip_scroll',

    'FORTNITE':'clip_scroll',
    'LEAGUE':'clip_scroll',
    'CSGO':'clip_scroll',
    'DOTA':'clip_scroll',
    'CLIPS':'clip_scroll',

    'CONNECTED_ACCOUNTS':'connected_accounts',
    'PROFILE':'own_profile_view',
    'ACCOUNT':'own_profile_view',
    'Login Page':'login',
    'LOGIN':'login',
    'Settings':'account_management',

    'UPGRADE':'upgrade',
    'STUDIO':'studio',
}

filtered_values = [
    'clip_scroll',
    'account_management',
    'own_profile_view',
    'user_scroll',
    'match-history',
    'clip_view',
    'dashboard',
    'studio',
    'iframe',
    'montage_scroll',
    'connected_accounts',
    'partner_claim',
    'upgrade',
    'remix',
    '404',
    'site_exploration',
    'discord_bot',
    ]

def get_month(x, frmt='%d.%m.%Y %H:%M'):
    if x is not type(dt):
        date = dt.strptime(x, frmt)
        return dt(date.year, date.month,1)
    else:
        return dt(x.year, x.month,1)

def get_date_elements(df, column):
    day = df[column].dt.day
    month = df[column].dt.month
    year = df[column].dt.year
    return day, month, year 

def get_users_appeared(df, more_than=1):

    user_counts = pd.DataFrame(df['User ID'].value_counts())
    users_appeared = list(user_counts[user_counts['count'] > more_than].index)

    return df[df['User ID'].isin(users_appeared)]

def clean_data(df, return_ids=False):

    data = df
    temp_df = pd.DataFrame()
    for val, sub_type in sub_zip:
        
        sub_df = data[(
                ~data['Simple checkout purchase amount'].isin(val)) &
                (data['Simple checkout purchase amount'] > 0) &
                (data['Subscription plan name'].isin(sub_dict[sub_type]))
            ][[ 
                'Transaction ID',
                'Date (GMT+3)', 
                'User ID', 
                'Simple checkout purchase amount', 
                'Subscription plan name']]
        
        temp_df = pd.concat([temp_df, sub_df])

        vals.extend(val)

    for val, sub_type in sub_zip:
        temp_df.loc[temp_df['Simple checkout purchase amount'].isin(val), 'Subscription plan name'] = sub_dict[sub_type][1]

    incorrect_ids = temp_df.loc[temp_df['Simple checkout purchase amount'].isin(vals)]['Transaction ID'].tolist()
    questionable_ids = temp_df.loc[~temp_df['Simple checkout purchase amount'].isin(vals)]['Transaction ID'].tolist()

    data = data.set_index('Transaction ID')
    temp_df = temp_df.set_index('Transaction ID')
    data.update(temp_df[temp_df.index.isin(incorrect_ids)])
    data = data[(~data.index.isin(questionable_ids)) & (data['Simple checkout purchase amount'] > 0)]

    if return_ids:
        return [data, incorrect_ids, questionable_ids]
    else:
        return data

def add_cohorts(df, id_col='User ID', date_col='Date (GMT+3)', date_fmt="%d.%m.%Y %H:%M"):

    # Turn Date+Time into Month for all Users and then store unique months for later use
    df['month'] = df[date_col].apply(lambda x: get_month(x=x, frmt=date_fmt)) 

    # Turn Date+Time into Month for all Users
    df['month'] = df[date_col].apply(lambda x: get_month(x=x, frmt=date_fmt))  
    # Find the first month a User appears and apply it to all instances of that User
    df['cohort_month'] =  df.groupby(id_col)['month'].transform('min')

    # Assign month/year that a User subscribed/resubscribed 
    _,sub_month,sub_year =  get_date_elements(df,'month')
    # Assign the cohort month/year from first subscription
    _,cohort_month,cohort_year =  get_date_elements(df,'cohort_month')

    # Determine difference between target subscription and first month of sub, and create index
    year_diff = sub_year -cohort_year
    month_diff = sub_month - cohort_month
    df['cohort_index'] = year_diff * 12 + month_diff + 1

    return df