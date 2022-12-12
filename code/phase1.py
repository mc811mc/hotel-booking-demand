# general imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# prevent warnings import
import warnings
warnings.filterwarnings("ignore")

# custom functions
## sample covariance matrix function
### the following function generates a heatmap
### of the sample covariance matrix
def covariance_matrix(dataset):
    plt.figure(figsize=(24,12))
    sns.set(font_scale=1.5)
    corr = dataset.corr()
    sns.heatmap(corr,
                cbar=True,
                cmap='RdBu_r',
                annot=True,
                linewidth=1,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                )
    plt.title("Covariance Matrix")
    plt.rc('axes', titlesize=50)
    plt.tight_layout()
    plt.show()

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

# prints out information used to create the dataset explanation table in the final project report
print(df.info())

# plots the heatmap for the correlation matrix of the preprocessed dataset
plt.figure(figsize=(40, 40))
sns.heatmap(df.corr(), cmap='crest')
plt.title('Correlation Matrix')
plt.xticks(rotation=60,ha='right')
plt.show()

#################
# data cleaning #
#################

# check dataset for missing values
print(f'Number of Missing Values per Column Before Data Cleaning: \n{df.isnull().sum()[df.isnull().sum() > 0]}\n')

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

# check dataset for missing values
print(f'Number of Missing Values per Column After Data Cleaning: \n{df.isnull().sum()[df.isnull().sum() > 0]}\n')

# this section handles binning countries into continents
# within the country of origin column.
# each country is categorized to one of the following: africa, asia, australia, europe, north america, south america,
# unknown, or others
## check the number of countries before binning to respective continents
print(f'Number of Continents Before Binning: {df.country.unique().size}')

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

# check to see if there are only continents
print(f'Number of Continents After Binning: {df.country.unique().size}')

# this section makes plots about the dataset
plt.rc('axes', titlesize=40)
plt.rc('axes', labelsize=30)

## plot hotel type booking ratio
plt.rcParams['figure.figsize'] = [25,25]
plt.subplot(2, 1, 1)
plt.pie(df.hotel.value_counts().values,
        labels=df.hotel.value_counts().index,
        autopct='%.2f%%',
        explode=[0, 0.1])
plt.title('Distribution of the type of Hotels')

## plot cancellation rate
plt.subplot(2, 1, 2)
crosstab_table=pd.crosstab(df.hotel, df.is_canceled, margins=True)
crosstab_table['cancel_percent']=crosstab_table[1] * 100 / crosstab_table['All']
crosstab_table.drop('All', axis = 0)['cancel_percent'].plot.bar()
plt.title('Cancellation Rate by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Cancellation %')
plt.show()

## plot arrival date month
plt.rcParams['figure.figsize']=[40, 40]
plt.subplot(2, 1, 1)
sns.countplot(x='arrival_date_month',
              data=df,
              order=df.arrival_date_month.value_counts().index)
plt.title('Number of Bookings by Month')
plt.xlabel('Booking Month')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=60,ha='right')

plt.subplot(2, 1, 2)
sns.countplot(x='arrival_date_month',
              data=df,
              order=df.arrival_date_month.value_counts().index,
              hue='hotel')
plt.title('Number of Bookings by Month and Hotel Type')
plt.xlabel('Booking Month')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=60,ha='right')
plt.show()

## plot cancellation by arrival date month
plt.figure(figsize=(25, 25))
crosstab_table=pd.crosstab(df.arrival_date_month, df.is_canceled, margins=True)
crosstab_table['cancel_percent']= crosstab_table[1] * 100 / crosstab_table['All']
crosstab_table.drop('All', axis=0)['cancel_percent'].plot.bar()
plt.title('Cancellation Percentage by Month')
plt.xlabel('Arrival Month')
plt.ylabel('Cancellation %')
plt.xticks(rotation=60,ha='right')
plt.show()

## plot booking made by continent
plt.figure(figsize=(40, 40))
plt.subplot(2, 1, 1)
sns.countplot(x='country',data=df)
plt.title('Booking by Continent')
plt.xlabel('Continent')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=60,ha='right')

## plot cancellation per continent
plt.subplot(2, 1, 2)
crosstab_table=pd.crosstab(df.country, df.is_canceled, margins=True)
crosstab_table['cancel_percent']= crosstab_table[1] * 100 / crosstab_table['All']
crosstab_table.drop('All', axis=0)['cancel_percent'].plot.bar()
plt.title('Cancellation Percentage by Continent')
plt.xlabel('Continent')
plt.ylabel('Cancellation %')
plt.xticks(rotation=60,ha='right')
plt.show()

## plot type of room reserved
plt.figure(figsize=(25, 25))
plt.subplot(2, 1, 1)
sns.countplot(x='reserved_room_type',data=df)
plt.title('Bookings by Reserved Room Type')
plt.xlabel('Reserved Room Type')
plt.ylabel('Number of Reserved Rooms')

## plot cancellation rate by type of room
plt.subplot(2, 1, 2)
crosstab_table=pd.crosstab(df.reserved_room_type,df.is_canceled,margins=True)
crosstab_table['cancel_percent']=crosstab_table[1]*100/crosstab_table['All']
crosstab_table.drop('All',axis=0)['cancel_percent'].plot.bar()
plt.title('Cancellation Percentage by Room Type')
plt.xlabel('Reserved Room Type')
plt.ylabel('Cancellation %')
plt.show()

## plot deposit type
plt.figure(figsize=(25, 25))
plt.subplot(2, 1, 1)
sns.countplot(x='deposit_type',data=df)
plt.title('Bookings by Deposit Type')
plt.xlabel('Deposit Type')
plt.ylabel('Total Number of Each Deposit Type')

## plot cancellation rate by deposit type
plt.subplot(2, 1, 2)
crosstab_table=pd.crosstab(df.deposit_type,df.is_canceled,margins=True)
crosstab_table['cancel_percent']=crosstab_table[1]*100/crosstab_table['All']
crosstab_table.drop('All',axis=0)['cancel_percent'].plot.bar()
plt.title('Cancellation Percentage by Deposit Type')
plt.xlabel('Deposit Type')
plt.ylabel('Cancellation %')
plt.xticks(rotation=60,ha='right')
plt.show()

# plot sample covariance matrix
covariance_matrix(df)

# choosing variables
correlation = df.corr()['is_canceled'].abs().sort_values(ascending = True)
print("Printing out Correlation")
print()
print(correlation)

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