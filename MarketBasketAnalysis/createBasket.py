
# coding: utf-8

# ## A program to create "basket" of items to perform Market Basket Analysis

# In[17]:

# Import necessary Libraries
import pandas as pd


# In[4]:

# Read data file
mkt = pd.read_csv("marketbasketData.csv")


# In[11]:

# check if data file is read
print(mkt.head())


# In[18]:

# create a basket 
var = mkt.apply(lambda x: x.isin([' true']), axis=1).apply(lambda x: list(mkt.columns[x]), axis=1)


# In[12]:

# create a dataframe to further process the data
df = pd.DataFrame(var)
print(df)


# In[13]:

# rename the column name
df.columns = ['Item']


# In[14]:

# convert the column to string
df['Item'] = [str(x) for x in df['Item']]
print(df)


# In[15]:

# strip the brackets in each row
df1 = df.Item.str.strip("[]")
print(df1)


# In[16]:

# write the dataframe to csv file without the index
df1.to_csv("output/basket.csv", index=False)

