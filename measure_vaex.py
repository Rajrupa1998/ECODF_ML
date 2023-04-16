import time
import vaex as ve
import pandas as pd1
from energy_measurement_main import measure_energy


# def check_counts(df):
#     df1 = ve.DataFrame()
#     df=ve.from_pandas(df1)
#     df.is

# I/O functions - READ
@measure_energy
def load_csv(path):
    return ve.read_csv(path)

@measure_energy
def load_hdf(path):
    return ve.open(path)

@measure_energy
def load_json(path):
    return ve.from_json(path)

# I/O functions - WRITE
@measure_energy
def save_csv(df, path):
    return df.export_csv(path)

@measure_energy
def save_hdf(df, path, key='a'):
    return df.export(path)

@measure_energy
def save_json(df, path):
    return df.export_json(path)

###------------------------------------------###

# Missing data handling 
@measure_energy
def dropna(df):
    df = pd1.DataFrame(df)
    df=ve.from_pandas(df)
    df.dropna()

@measure_energy
def fillna(df, val):
    df = pd1.DataFrame(df)
    df=ve.from_pandas(df)
    return df.fillna(val)

@measure_energy
def isna(df, cname):
    return df[cname].isna()

@measure_energy
def replace(df, cname, src, dest):
    return df[cname].str.replace(src, dest)

###------------------------------------------###

# Table operations
# drop column
# groupby
# merge 
# transpose
# sort


@measure_energy
def drop(df, col_names=[]):
    df.drop(columns=col_names)

@measure_energy
def groupby(df, cname):
    return df.groupby(cname)

@measure_energy
def merge(df1, df2, on=None, lsuffix="_l"):
    if(on == None):
        return df1.join(df2, lsuffix="_l")
    else:
        return df1.join(df2, on=on, how='inner', lsuffix=lsuffix)

@measure_energy
def sort(df, cname):
    return df.sort(cname)

@measure_energy
def concat_dataframes(df1, df2):
    return ve.concat([df1, df2])

###--------------------------------------------###
# Statistical Operations
# min, max, mean, count, unique, correlation

# count 
@measure_energy
def count(df):
    df1 = pd1.DataFrame()
    df=ve.from_pandas(df1)
    return df.count()

# sum
@measure_energy
def sum(df, cname):
    return df.sum(cname)

# mean
@measure_energy
def mean(df):
    df1 = pd1.DataFrame()
    df=ve.from_pandas(df1)
    return df.mean()

# min
@measure_energy
def min(df):
    
    df = pd1.DataFrame(df)
    df=ve.from_pandas(df)
    df['min']=df.min(binby='min')

@measure_energy
def max(df):
    df1 = pd1.DataFrame()
    df=ve.from_pandas(df1)
    return df.max()

@measure_energy
def unique(df):
    df1 = pd1.DataFrame()
    df=ve.from_pandas(df1)
    return df.unique()

# # subset 
# @measure_energy(handler=csv_handler)
# def subset(df, subset):
#     return df[subset]

# @measure_energy(handler=csv_handler)
# def sample(df, cnt=1000):
#     return df.sample(cnt)


# df = load_csv('../Datasets/adult.csv')
# drop_column(df, col_names=['age', 'education', 'occupation'])

# dfa = ve.from_csv('../Datasets/adult.csv')
# dfb = ve.from_csv('../Datasets/adult.csv')
# concat_dataframes(dfa, dfb)


# SUBSET = ['age', 'workclass', 'education', 'sex', 'race']
# SUBSET_A = ['occupation', 'relationship']
# subset(df, SUBSET)

# sample(df, 1000)
# sample(df, 10000)
# sample(df, 20000)


# col = ['capital.gain', 'capital.loss', 'hours.per.week']
# # col = ['capital.gain', 'capital.loss']
# sum(df[col])
# mean(df, 'age')
# min(df['capital.gain'])
# max(df['capital.gain'])
# unique(df['age'])
# print("Starting Adult Vaex Process...")
# for i in  range(10):
#     # Input Output functions
#     df = load_csv(path='../Datasets/adult.csv')
#     sleep()
#     df = load_json(path='../Datasets/adult.json')
#     sleep()
#     df = load_hdf(path='../Datasets/adult_v.hdf5')
#     sleep()

#     save_csv(df, f'df_adult_vaex_{i}.csv')
#     sleep()
#     save_json(df, f'df_adult_vaex_{i}.json')
#     sleep()
#     save_hdf(df, f'df_adult_vaex_{i}.hdf5')
#     sleep()

#     # --------------------------------------------------

#     # Handling missing data
#     df = ve.read_csv('../Datasets/adult.csv')
#     sleep()
#     isna(df, cname='workclass')
#     sleep()
#     dropna(df)
#     sleep()
#     fillna(df, val='0')
#     sleep()
#     replace(df, cname='workclass', src='?', dest='X')
#     sleep()
#     #
#     # --------------------------------------------------
#     # Table operations
#     df = ve.read_csv('../Datasets/adult.csv')
#     sleep()
#     df_samp = ve.read_csv('../Datasets/adult.csv')
#     sleep()
#     drop(df, col_names=['age', 'education'])
#     sleep()
#     groupby(df, cname='workclass')
#     sleep()

#     concat_dataframes(df, df_samp)
#     sleep()

#     sort(df, 'age')
#     sleep()
#     merge(df, df_samp)
#     sleep()

#     # ------------------------------------------
#     # Statistical operations
#     df = ve.read_csv('../Datasets/adult.csv')
#     sleep()
#     count(df)
#     sleep()
#     sum(df, 'capital.gain')
#     sleep()
#     mean(df['age'])
#     sleep()
#     min(df['capital.gain'])
#     sleep()
#     max(df['capital.gain'])
#     sleep()
#     unique(df['age'])
#     sleep()

#     print(f"finished iteration {i+1}")

# print("Process Complete...")

# csv_handler.save_data()