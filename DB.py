#!/usr/bin/env python
# coding: utf-8

# In[3]:


import duckdb

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# In[4]:


df = pd.read_csv("poverty_raw_data.csv")
df.head(10)


# In[6]:


query1 = duckdb.query('''SELECT
            country_name,
            survey_acronym,
            survey_year
            FROM df
            WHERE region_code ='LAC' and
            survey_coverage = 'national'
            GROUP BY 
            country_name,
            survey_acronym,
            survey_year;
            ''')
query1


# In[8]:


query1 = duckdb.query('''SELECT
            country_name,
            AVG(reporting_pce) AS min_avg_reporting_pce
            FROM df
            WHERE region_code ='SSA' 
            GROUP BY 
            country_name
            HAVING
            AVG(reporting_pce) <= ALL 
                (SELECT avg_reporting_pce FROM
                    (SELECT
                    country_name,
                    AVG(reporting_pce) AS avg_reporting_pce
                    FROM df
                    WHERE region_code ='SSA'
                    GROUP BY 
                    country_name)
                    WHERE avg_reporting_pce IS NOT NULL)''')
query1


# In[5]:


df.info()


# In[6]:


df[df["region_code"].isna()]


# Approaches for Dealing with NULLs
# 
# Ignoring them
# 
# Deleting the rows containing them (listwise deletion)
# 
# Imputation based on theencie)scolumn
# 
# Imputation base on othe
# (Imputation Based on Functional
# Dependencie)r columns

# In[7]:


region_codes = df[~df['region_code'].isna()].set_index("country_code")['region_code'].to_dict()

region_country_codes = df[df['region_code'].isna()]['country_code'].map(region_codes)
df['region_code'] = df['region_code'].fillna(region_country_codes)


# In[ ]:





# In[8]:


df.loc[df['country_name'] == 'Gabon', 'region_code'] = df.loc[df['country_name'] == 'Gabon', 'region_code'].fillna('SSA')
df.loc[df['country_name'] == 'Guyana', 'region_code'] = df.loc[df['country_name'] == 'Guyana', 'region_code'].fillna('LAC')


# In[9]:


df.groupby("survey_coverage")["record_id"].count()


# Simple Univariate Imputation: Most frequent value observed in the column

# In[10]:


df["survey_coverage"] = df["survey_coverage"].fillna("national")


# In[11]:


df["reporting_pce"] = df.groupby("country_code")["reporting_pce"].transform(lambda x: x.fillna(x.mean()))


# In[12]:


df["reporting_pce"] = df.groupby("region_code")["reporting_pce"].transform(lambda x: x.fillna(x.mean()))


# In[13]:


duckdb.query('''SELECT
            country_name,
            survey_acronym,
            survey_year
            FROM df
            WHERE region_code ='LAC' and
            survey_coverage = 'national'
            GROUP BY 
            country_name,
            survey_acronym,
            survey_year
            ORDER BY country_name, survey_year
            ''')


# In[15]:


duckdb.query('''SELECT
            country_name,
            AVG(reporting_pce) AS min_avg_reporting_pce
            FROM df
            WHERE region_code ='SSA' 
            GROUP BY 
            country_name
            HAVING
            AVG(reporting_pce) <= ALL 
                (SELECT avg_reporting_pce FROM
                    (SELECT
                    country_name,
                    AVG(reporting_pce) AS avg_reporting_pce
                    FROM df
                    WHERE region_code ='SSA'
                    GROUP BY 
                    country_name)
                    WHERE avg_reporting_pce IS NOT NULL)''')


# 1. country_code- country_name
# 2. country_code - region_code

# In[16]:


countries = df[["country_code", "country_name"]].drop_duplicates()
regions = df[["country_code", "region_code"]].drop_duplicates()
df = df.drop(["country_name", "region_code"], axis=1)


# In[17]:


duckdb.query('''SELECT
            C.country_name,
            survey_acronym,
            survey_year
            FROM df, regions AS R, countries AS C
            WHERE df.country_code = R.country_code AND
                df.country_code = C.country_code AND
                R.region_code ='LAC' AND
                survey_coverage = 'national'
            GROUP BY 
            country_name,
            survey_acronym,
            survey_year
            ORDER BY C.country_name, survey_year
            ''')


# In[25]:


duckdb.query('''SELECT
            C.country_name,
            AVG(reporting_pce) AS min_avg_reporting_pce
            FROM df, regions AS R, countries AS C
            WHERE df.country_code = R.country_code AND R.region_code ='SSA' AND
                df.country_code = C.country_code
            GROUP BY 
            C.country_name
            HAVING
            AVG(reporting_pce) <= ALL 
                (SELECT avg_reporting_pce FROM
                    (SELECT
                    df.country_code,
                    AVG(reporting_pce) AS avg_reporting_pce
                    FROM df, regions AS R
                    WHERE df.country_code = R.country_code AND R.region_code ='SSA'
                    GROUP BY 
                    df.country_code)
                    WHERE avg_reporting_pce IS NOT NULL)''')


# In[ ]:




