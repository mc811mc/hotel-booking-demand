# general imports
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split

# regression analysis imports
import scipy.stats as st
from sklearn import metrics
import statsmodels.api as sm

# prevent warnings import
import warnings
warnings.filterwarnings("ignore")

# custom functions
## function to bin each country to their continent
### the following function performs binning
### from country to their respective continent
def country_to_continent(country):
    if country in africa:
        return 'Africa'
    elif country in asia:
        return 'Asia'
    elif country in australia:
        return 'Australia'
    if country in europe:
        return 'Europe'
    elif country in north_america:
        return 'North America'
    elif country in south_america:
        return 'South America'
    elif country in unk:
        return 'Unknown'
    elif country in Others:
        return 'Others'

# import data
original_csv = 'https://raw.githubusercontent.com/mc811mc/hotel-booking-demand/main/dataset/hotel_bookings.csv'

data = pd.read_csv(original_csv)
df = data.copy()

#################
# data cleaning #
#################

# fill missing values in the children column with '0'
df.children.fillna(0,inplace=True)

# fill missing values in the country column with 'Unknown'
df.country.fillna('Unknown',inplace=True)

# replace agent ids (numerical) values in the agent column with 'Agent' (categorical)
df.loc[df.agent.isnull() == False,'agent'] = 'Agent'

# fill null values in the agent column with 'No agent'
df.agent.fillna('No agent',inplace = True)

# put in 'Corporate' in the company column
# if market segment and distribution channel column is both 'Corporate'
df.loc[((df.market_segment == 'Corporate') | (df.distribution_channel == 'Corporate')) & (df.company.isnull()), 'company'] ='Corporate'

# replace non-null values in company column as 'Corporate'
df.loc[df.company.isnull() == False,'company'] = 'Corporate'

# fill in remaining and missing values in the company column with 'Individuals'
df.company.fillna('Individuals',inplace=True)

# correct the data type of the column arrival date year
df.arrival_date_year = df.arrival_date_year.astype(object)

# drop all duplicate rows in the dataset
df.drop_duplicates(inplace = True)

# this section handles binning countries into continents
# within the country of origin column.
# each country is categorized to one of the following: africa, asia, australia, europe, north america, south america,
# unknown, or others

## binning country column to its respective continents
africa = ['MOZ','BWA','MAR','ZAF','AGO','ZMB','ZWE','DZA','TUN','CAF','NGA','SEN','SYC','CMR','MUS','COM','UGA','CIV',
       'BDI','EGY','MWI','MDG','TGO','DJI','STP','ETH','RWA','BEN','TZA','GHA','KEN','GNB','BFA','LBY','MLI','NAM',
       'MRT','SDN','SLE']

asia = ['OMN','CN','IND','CHN','ISR','KOR','ARE','HKG','IRN','CYP','KWT','MDV','KAZ','PAK','IDN','LBN','PHL','AZE','BHR',
     'THA','MYS','ARM','JPN','LKA','JOR','SYR','SGP','SAU','VNM','QAT','UZB','NPL','MAC','TWN','IRQ','KHM','BGD','TJK',
     'TMP','MMR','LAO']

australia = ['AUS']

europe = ['PRT','GBR','ESP','IRL','FRA','ROU','NOR','POL','DEU','BEL','CHE','GRC','ITA','NLD','DNK','RUS','SWE','EST',
       'CZE','FIN','LUX','SVN','ALB','UKR','SMR','LVA','SRB','AUT','BLR','LTU','TUR','HUN','HRV','GEO','AND','SVK',
       'MKD','BIH','BGR','MLT','ISL','MCO','LIE','MNE']

north_america = ['USA', 'MEX', 'PRI', 'CRI', 'CUB', 'HND', 'NIC', 'GAB', 'PAN', 'SLV', 'GTM']

south_america = ['ARG', 'BRA', 'CHL', 'URY', 'COL', 'VEN', 'SUR', 'PER', 'ECU', 'BOL', 'PRY', 'GUY']

unk = ['Unknown']

Others = ['CYM','CPV','JAM','GIB','JEY','GGY','FJI','NZL','DOM','PLW','BHS','KNA','IMN','VGB','GLP','UMI','MYT','FRO',
       'BRB','ABW','AIA','DMA','PYF','LCA','ATA','ASM','NCL','KIR','ATF']

# modify country column to make only continents appear
df.country = df.country.apply(country_to_continent)

# dimensionality reduction
## create total_members column that includes children, babies, and adults
df['total_members'] = df.children + df.babies + df.adults

## create total nights column that includes stays in week nights and stays in weekend nights columns
df['total_nights'] = df.stays_in_week_nights + df.stays_in_weekend_nights

## drop the aggregated columns in the prior steps
df.drop(columns=['children', 'babies', 'adults', 'stays_in_week_nights', 'stays_in_weekend_nights'], inplace=True)

## drop irrelevant columns to improve model
df.drop(columns=['days_in_waiting_list', 'reservation_status','reservation_status_date'],inplace=True)

# feature encoding
df['hotel'] = df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})

df['meal'] = df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})

df['reserved_room_type'] = df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                   'L': 7, 'P': 8, 'B': 9})
df['deposit_type'] = df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})

# features to be dropped through regression analysis
df.drop(columns=['country', 'arrival_date_year', 'market_segment', 'agent', 'company'],inplace=True)
df.drop(df.loc[df['assigned_room_type']=='L'].index, inplace=True)
df.drop(df.loc[df['assigned_room_type']=='P'].index, inplace=True)
df.drop(df.loc[df['distribution_channel']=='Undefined'].index, inplace=True)

# feature selection
## get rid of rows that have no guests
df_temp=df.loc[~((df.total_members==0) & (df.is_canceled==0))]

## get rid of rows that have no number of nights stayed
df_temp=df_temp.loc[~((df_temp.total_nights==0) & (df_temp.is_canceled==0))]

df_preprocessed = pd.get_dummies(df_temp, drop_first=True)

# attribute variables
X = df_preprocessed.drop(columns='is_canceled')

# target variable
y = df_preprocessed.is_canceled

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# association analysis, f-test analysis, stepwise regression analysis, collinearity analysis
## perform ols
model_ols = sm.OLS(y_train, X_train).fit()
print(model_ols.summary())
y_pred_ols = model_ols.predict(X_test)

print(f'Mean Absolute Error: {metrics.mean_squared_error(y_test, y_pred_ols)}')