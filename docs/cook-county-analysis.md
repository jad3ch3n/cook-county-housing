# Cook County Housing Equity Analysis

**Table of Contents**  
1. [Overview](#1-overview)
2. [Data Dictionary](#2-data-dictionary)
3. [Cleaning & Exploratory Data Analysis](#3-cleaning--exploratory-data-analysis)
4. [Data Processing Pipeline & Modeling](#4-data-processing-pipeline--modeling)
5. [Evaluation & Metrics](#5-evaluation--metrics)
6. [Conclusions & Next Steps](#6-conclusions--next-steps)
---

## 1. Overview

This project investigates 2013 to 2019 housing sale data from Cook County with a dual focus: building predictive models for property valuation and examining the fairness of those models in real-world contexts. Using exploratory analysis and visual diagnostics, I identified both the key features that drive sale prices and the systemic patterns that lead to inequitable outcomes in tax assessments. I demonstrate the full data science lifecycle to model a substantial dataset comprising over 200,000 training observations and 68,000 testing observations.

* **Objective:** Analyze housing price disparities across Cook County neighborhoods.
* **Data Source:** Cook County Assessor’s Office property records.
* **Approach:** Data cleaning, EDA, regression modeling, subgroup evaluation.


**Imports Setup**


```python
# Standard imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lm

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Project-specific helper functions
from feature_func_inline import *  # This loads remove_outliers, log_transform, etc.

# Plot settings
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12
```

---

## 2. Data Dictionary

*See separate `codebook.txt` for full definitions.*  
Key variables:  
- `Sale Price` (float): Final sale price in USD.  
- `Log Sale Price` (float): Natural log of sale price.  
- `Neighborhood Code` (int): Categorical code (1–77) for areas.


```python
# Display the codebook for reference
print(open('data/codebook.txt').read())
```

    Fomat: Each column in the dataframe is documented in 3 consecutive rows (column label, description, data type).
    ----------------------------------------------------------------------------------------------------------------------------------
    PIN
    Unique Permanent Identification Number for each property. All PINs are 14 digits: 2 digits for area + 2 digits for sub area + 2 digits for block + 2 digits for parcel + 4 digits for the multicode
    Plain Text
    
    
    Property Class
    Property class. For a list of property classes, see https://www.cookcountyassessor.com/assets/forms/classcode.pdf
    Number
    
    
    Neighborhood Code
    Neighborhood code as assigned by the Assessment office. An interactive map is available at https://maps.cookcountyil.gov/cookviewer/
    Number
    
    
    Land Square Feet
    Square feet of the land (not just the building) of the property. Note that land is divided into 'plots' and 'parcels' - this field applies to parcels, identified by PIN
    Number
    
    
    Town Code
    Township code as assigned by the Assessment office. An interactive map is available at https://maps.cookcountyil.gov/cookviewer/
    Number
    
    
    Apartments
    Number of apartments in the building - 2 = Two, 3 = Three, 4 = Four, 5 = Five, 6 = Six, 0 = None
    Number
    
    
    Wall Material
    Exterior wall material - 1=Wood, 2=Masonry, 3=Wood&Masonry, 4=Stucco
    Number
    
    
    Roof Material
    Roof construction material. 1 = Shingle/Asphalt, 2 = Tar & Gravel, 3 = Slate, 4 = Shake, 5 = Tile, 6 = Other
    Number
    
    
    Basement
    Basement type - 1 = Full, 2 = Slab, 3 = Partial, 4 = Crawl
    Number
    
    
    Basement Finish
    Basement finish - 1 = Formal rec room; 2 = Apartment; 3 = Unfinished
    Number
    
    
    Central Heating
    Central heating type - 1 = Warm air, 2 = Hot water steam, 3 = Electric, 4 = Other
    Number
    
    
    Other Heating
    Other heating type - 1 = Floor furnace, 2 = Unit heater, 3 = Stove, 4 = Solar, 5 = none
    Number
    
    
    Central Air
    Is central airconditioning present? - 1 = yes, 2 = no
    Number
    
    
    Fireplaces
    Number of fireplaces, counted as the number of flues one can see from the outside of the building.
    Number
    
    
    Attic Type
    Type of attic - 1 = Full, 2 = partial, 3 = none
    Number
    
    
    Attic Finish
    Finish of attic - 1 = Living area, 2 = Apartment, 3 = unfinished
    Number
    
    
    Design Plan
    Plan of Design - 1 = architect, 2 = stock plan
    Number
    
    
    Cathedral Ceiling
    Cathedral Ceiling - 1 = yes, 2 = No. (Note: the column name comes from a variable that is no longer used, but the column name was not changed to reflect Cathedral Ceiling)
    Number
    
    
    Construction Quality
    Construction quality - 1 = Deluxe, 2 = Average, 3 = Poor (There is one 4 in this dataset, but no 3. In general, this column is not useful for analytical purposes)
    Number
    
    
    Site Desirability
    Site desirability - 1 = Beneficial to Value, 2 = Not relevant to Value, 3 = Detracts from Value. This field lack sufficient variation to be useful for modeling.
    Number
    
    
    Garage 1 Size
    Garage 1 size - 1 = 1 car, 2 = 1.5 car, 3 = 2 car, 4 = 2.5 cars, 5 = 3 cars, 6 = 3.5 cars, 7 = none, 8 = 4 cars
    Number
    
    
    Garage 1 Material
    Garage 1 construction - 1 = Frame, 2 = Masonry, 3= Frame/Masonry, 4 = Stucco
    Number
    
    
    Garage 1 Attachment
    Is Garage 1 attached? 1 = Yes, 2 = No
    Number
    
    
    Garage 1 Area
    Is Garage 1 physically including within the building area? 1 = Yes, 2 = No. If yes, the garage area is subtracted from the building square feet calculation by the field agent.
    Number
    
    
    Garage 2 Size
    Garage 2 size - 1 = 1 car, 2 = 1.5 car, 3 = 2 car, 4 = 2.5 cars, 5 = 3 cars, 6 = 3.5 cars, 7 = none, 8 = 4 cars
    Number
    
    
    Garage 2 Material
    Garage 2 construction - 1 = Frame, 2 = Masonry, 3= Frame/Masonry, 4 = Stucco
    Number
    
    
    Garage 2 Attachment
    Is Garage 2 attached? 1 = Yes, 2 = No
    Number
    
    
    Garage 2 Area
    Is Garage 2 physically including within the building area? 1 = Yes, 2 = No. If yes, the garage area is subtracted from the building square feet calculation by the field agent.
    Number
    
    
    Porch
    Enclosed porch - 1 = Frame, 2 = Masonry, 3 = None
    Number
    
    
    Other Improvements
    Other improvements - if not 0, they contain a code. The definitions of these codes are maintained by the field workers. Unfortunatly, the way this data is entered makes it impossible to distinguish between a 1 & 2 and a 12, making this field mostly useless.
    Number
    
    
    Building Square Feet
    Building square feet, as measured from the exterior of the property
    Number
    
    
    Repair Condition
    State of Repair - 1 = Above average, 2 = Average, 3 = Below average
    Number
    
    
    Multi Code
    Variable that indicates that more one building exists on the PIN. 2 = one building, 3 = two buildings, etc… through 7 = 6 buildings.
    Number
    
    
    Number of Commercial Units
    Number of commercial units (the vast majority are for properties with class 212).
    Number
    
    
    Estimate (Land)
    Board of Review final estimated market value of land from the prior tax year.
    Number
    
    
    Estimate (Building)
    Board of Review final estimated market value of building from the prior tax year.
    Number
    
    
    Deed No.
    Deed number for sale
    Plain Text
    
    
    Sale Price
    Sale price
    Number
    
    
    Longitude
    Longitude coordinate of the property's location, as defined by the centroid of the parcel shape in GIS.
    Number
    
    
    Latitude
    Latitude coordinate of the property's location, as defined by the centroid of the parcel shape in GIS.
    Number
    
    
    Census Tract
    Census tract identifier - full map available through the Census Bureau, https://www.census.gov/geo/maps-data/maps/2010ref/st17_tract.html
    Plain Text
    
    
    Multi Property Indicator
    Indicator for a property with multiple improvements on one PIN, e.g. a main house and a coach house. NOT to be confused with a property which was part of a multi-pin sale.
    Number
    
    
    Modeling Group
    Modeling group, as defined by the property class. Properties with class 200, 201, 241, 299 is defined as "NCHARS", short for "no characteristics", which are condos and vacant land classes. Properties with class 202, 203, 204, 205, 206, 207, 208, 209, 210, 235, 278, and 295 are "SF", short for "single-family." Properties with class 211 and 212 are "MF", short for "multi-family."
    Plain Text
    
    
    Age
    Age of the property. If missing, this defaults to 10. This field is a combination of original age and effective age where original age refers to the oldest component of the building and effective age is a relative judgement due to renovations or other improvements. For instance, if a property is completely demolished and built up again, the age resets to 1. But if portions of the original structure are kept, it may be more complicated to determine the age.
    Number
    
    
    Use
    Use of property - 1 = single family, 2 = multi-family
    Number
    
    
    O'Hare Noise
    Indicator for the property under O'Hare approach flight path, within 1/4 mile.
    Number
    
    
    Floodplain
    Indicator for properties on a floodplain, defined as a FEMA Special Flood Hazard Area
    Number
    
    
    Road Proximity
    Indicates whether the property is within 300 ft of a major road.
    Number
    
    
    Sale Year
    Year of sale
    Number
    
    
    Sale Quarter
    Quarter of sale
    Number
    
    
    Sale Half-Year
    Half-year of sale
    Number
    
    
    Sale Quarter of Year
    Quarter of year of sale
    Number
    
    
    Sale Month of Year
    Month of year of sale
    Number
    
    
    Sale Half of Year
    Half of year of sale
    Number
    
    
    Most Recent Sale
    Indicator that this sale is the most recent sale of the property.
    Number
    
    
    Age Decade
    Age in decades
    Number
    
    
    Pure Market Filter
    Indicator for pure market sale
    Number
    
    
    Garage Indicator
    Indicates presence of a garage of any size.
    Number
    
    
    Neigborhood Code (mapping)
    Unmodified neighborhood code that can be used for mapping.
    Number
    
    
    Town and Neighborhood
    Combination of town and neighborhood used to uniquely neighborhoods across townships.
    Number
    
    
    Description
    Short description of the household.
    Plain Text
    

---

## 3. Cleaning & Exploratory Data Analysis
### Data Cleaning & Preprocessing Pipeline
1. Raw load & sanity checks
2. Missing-value handling
3. Infinite values handling
4. Outlier treatment
5. Type conversions
6. Categorical encoding
7. Feature scaling
8. Final checks


```python
training_data = pd.read_csv("data/cook_county_train.csv", index_col='Unnamed: 0')
```

As a good sanity check, I verify that the data shape matches the description.


```python
# 204,792 observations and 62 features in training data
assert training_data.shape == (204792, 62)
# Sale Price is provided in the training data
assert 'Sale Price' in training_data.columns.values
```


```python
training_data.columns.values
```




    array(['PIN', 'Property Class', 'Neighborhood Code', 'Land Square Feet',
           'Town Code', 'Apartments', 'Wall Material', 'Roof Material',
           'Basement', 'Basement Finish', 'Central Heating', 'Other Heating',
           'Central Air', 'Fireplaces', 'Attic Type', 'Attic Finish',
           'Design Plan', 'Cathedral Ceiling', 'Construction Quality',
           'Site Desirability', 'Garage 1 Size', 'Garage 1 Material',
           'Garage 1 Attachment', 'Garage 1 Area', 'Garage 2 Size',
           'Garage 2 Material', 'Garage 2 Attachment', 'Garage 2 Area',
           'Porch', 'Other Improvements', 'Building Square Feet',
           'Repair Condition', 'Multi Code', 'Number of Commercial Units',
           'Estimate (Land)', 'Estimate (Building)', 'Deed No.', 'Sale Price',
           'Longitude', 'Latitude', 'Census Tract',
           'Multi Property Indicator', 'Modeling Group', 'Age', 'Use',
           "O'Hare Noise", 'Floodplain', 'Road Proximity', 'Sale Year',
           'Sale Quarter', 'Sale Half-Year', 'Sale Quarter of Year',
           'Sale Month of Year', 'Sale Half of Year', 'Most Recent Sale',
           'Age Decade', 'Pure Market Filter', 'Garage Indicator',
           'Neigborhood Code (mapping)', 'Town and Neighborhood',
           'Description', 'Lot Size'], dtype=object)




```python
training_data['Description'][0]
```




    'This property, sold on 09/14/2015, is a one-story houeshold located at 2950 S LYMAN ST.It has a total of 6 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.'




```python
training_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PIN</th>
      <th>Property Class</th>
      <th>Neighborhood Code</th>
      <th>Land Square Feet</th>
      <th>Town Code</th>
      <th>Apartments</th>
      <th>Wall Material</th>
      <th>Roof Material</th>
      <th>Basement</th>
      <th>Basement Finish</th>
      <th>...</th>
      <th>Sale Month of Year</th>
      <th>Sale Half of Year</th>
      <th>Most Recent Sale</th>
      <th>Age Decade</th>
      <th>Pure Market Filter</th>
      <th>Garage Indicator</th>
      <th>Neigborhood Code (mapping)</th>
      <th>Town and Neighborhood</th>
      <th>Description</th>
      <th>Lot Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17294100610000</td>
      <td>203</td>
      <td>50</td>
      <td>2500.0</td>
      <td>76</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>9</td>
      <td>2</td>
      <td>1.0</td>
      <td>13.2</td>
      <td>0</td>
      <td>0.0</td>
      <td>50</td>
      <td>7650</td>
      <td>This property, sold on 09/14/2015, is a one-story houeshold located at 2950 S LYMAN ST.It has a total of 6 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>2500.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13272240180000</td>
      <td>202</td>
      <td>120</td>
      <td>3780.0</td>
      <td>71</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>9.6</td>
      <td>1</td>
      <td>1.0</td>
      <td>120</td>
      <td>71120</td>
      <td>This property, sold on 05/23/2018, is a one-story houeshold located at 2844 N LOWELL AVE.It has a total of 6 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>3780.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25221150230000</td>
      <td>202</td>
      <td>210</td>
      <td>4375.0</td>
      <td>70</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>11.2</td>
      <td>1</td>
      <td>1.0</td>
      <td>210</td>
      <td>70210</td>
      <td>This property, sold on 02/18/2016, is a one-story houeshold located at 11415 S PRAIRIE AVE.It has a total of 7 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>4375.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10251130030000</td>
      <td>203</td>
      <td>220</td>
      <td>4375.0</td>
      <td>17</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>7</td>
      <td>2</td>
      <td>1.0</td>
      <td>6.3</td>
      <td>1</td>
      <td>1.0</td>
      <td>220</td>
      <td>17220</td>
      <td>This property, sold on 07/23/2013, is a one-story with partially livable attics houeshold located at 2012 DOBSON ST.It has a total of 5 rooms, 3 of which are bedrooms, and 1.5 of which are bathrooms.</td>
      <td>4375.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31361040550000</td>
      <td>202</td>
      <td>120</td>
      <td>8400.0</td>
      <td>32</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>6</td>
      <td>1</td>
      <td>0.0</td>
      <td>6.3</td>
      <td>1</td>
      <td>1.0</td>
      <td>120</td>
      <td>32120</td>
      <td>This property, sold on 06/10/2016, is a one-story houeshold located at 104 SAUK TRL.It has a total of 5 rooms, 2 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>8400.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 62 columns</p>
</div>



Each row represents a sale of a property in Cook County, which contains attributes of the property such as lot size, neighborhood code, etc.

I think this data was collected to track the changes that has occured to the properties in Cook County. One could analyze property market price trends, track renovations to the wall, roof, basement, garage, and even see how often properties are being sold to understand the general property market selling/buying behaviors. This is information useful to the city, investors in real estate, and the average buyer interested in gauging when to buy or sell.


```python
training_data['Sale Month of Year'].mode()
```




    0    8
    Name: Sale Month of Year, dtype: int64




```python
# Check sale years in training data
years = training_data['Sale Year'].dropna().unique()
years = sorted(years)
print("Sale Years in training data:", years)
```

    Sale Years in training data: [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    

I question whether older people buy older properties. To answer this, I create a scatter plot using the "Age Decade" colum of training_data and the age column of the demographic dataset. I check whether there is any pattern or concentration in the scatterplot visually then perhaps map a KDE plot to further study the density.


```python
def plot_distribution(data, label):
    fig, axs = plt.subplots(nrows=2)

    sns.distplot(
        data[label], 
        ax=axs[0]
    )
    sns.boxplot(
        x=data[label],
        width=0.3, 
        ax=axs[1],
        showfliers=False,
    )

    # Align axes
    spacer = np.max(data[label]) * 0.05
    xmin = np.min(data[label]) - spacer
    xmax = np.max(data[label]) + spacer
    axs[0].set_xlim((xmin, xmax))
    axs[1].set_xlim((xmin, xmax))

    # Remove some axis text
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)

    # Put the two plots together
    plt.subplots_adjust(hspace=0)
    fig.suptitle("Distribution of " + label)
```


```python
plot_distribution(training_data, label='Sale Price')
```


    
![png](cook-county-analysis_files/cook-county-analysis_21_0.png)
    


I also take a look at some descriptive statistics of this variable.


```python
training_data['Sale Price'].describe()
```




    count    2.047920e+05
    mean     2.451646e+05
    std      3.628694e+05
    min      1.000000e+00
    25%      4.520000e+04
    50%      1.750000e+05
    75%      3.120000e+05
    max      7.100000e+07
    Name: Sale Price, dtype: float64



The visualization is skewed to the left by a lot, which could be a result of outliers embodying extraordinarily expensive properties. I can simply remove the outliers, but this is bad practice since it jeapardizes data integrity. Instead, I  rescale the data spread through log transformations to fix the skewness.


```python
training_data = training_data[training_data['Sale Price'] >= 500]
training_data['Log Sale Price'] = np.log(training_data['Sale Price'])
training_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PIN</th>
      <th>Property Class</th>
      <th>Neighborhood Code</th>
      <th>Land Square Feet</th>
      <th>Town Code</th>
      <th>Apartments</th>
      <th>Wall Material</th>
      <th>Roof Material</th>
      <th>Basement</th>
      <th>Basement Finish</th>
      <th>...</th>
      <th>Sale Half of Year</th>
      <th>Most Recent Sale</th>
      <th>Age Decade</th>
      <th>Pure Market Filter</th>
      <th>Garage Indicator</th>
      <th>Neigborhood Code (mapping)</th>
      <th>Town and Neighborhood</th>
      <th>Description</th>
      <th>Lot Size</th>
      <th>Log Sale Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13272240180000</td>
      <td>202</td>
      <td>120</td>
      <td>3780.0</td>
      <td>71</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1.0</td>
      <td>9.6</td>
      <td>1</td>
      <td>1.0</td>
      <td>120</td>
      <td>71120</td>
      <td>This property, sold on 05/23/2018, is a one-story houeshold located at 2844 N LOWELL AVE.It has a total of 6 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>3780.0</td>
      <td>12.560244</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25221150230000</td>
      <td>202</td>
      <td>210</td>
      <td>4375.0</td>
      <td>70</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0</td>
      <td>11.2</td>
      <td>1</td>
      <td>1.0</td>
      <td>210</td>
      <td>70210</td>
      <td>This property, sold on 02/18/2016, is a one-story houeshold located at 11415 S PRAIRIE AVE.It has a total of 7 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>4375.0</td>
      <td>9.998798</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10251130030000</td>
      <td>203</td>
      <td>220</td>
      <td>4375.0</td>
      <td>17</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2</td>
      <td>1.0</td>
      <td>6.3</td>
      <td>1</td>
      <td>1.0</td>
      <td>220</td>
      <td>17220</td>
      <td>This property, sold on 07/23/2013, is a one-story with partially livable attics houeshold located at 2012 DOBSON ST.It has a total of 5 rooms, 3 of which are bedrooms, and 1.5 of which are bathrooms.</td>
      <td>4375.0</td>
      <td>12.323856</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31361040550000</td>
      <td>202</td>
      <td>120</td>
      <td>8400.0</td>
      <td>32</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>0.0</td>
      <td>6.3</td>
      <td>1</td>
      <td>1.0</td>
      <td>120</td>
      <td>32120</td>
      <td>This property, sold on 06/10/2016, is a one-story houeshold located at 104 SAUK TRL.It has a total of 5 rooms, 2 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>8400.0</td>
      <td>10.025705</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30314240080000</td>
      <td>203</td>
      <td>181</td>
      <td>10890.0</td>
      <td>37</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2</td>
      <td>1.0</td>
      <td>10.9</td>
      <td>1</td>
      <td>1.0</td>
      <td>181</td>
      <td>37181</td>
      <td>This property, sold on 10/26/2017, is a one-story with partially livable attics houeshold located at 2820 186TH ST.It has a total of 6 rooms, 4 of which are bedrooms, and 1.5 of which are bathrooms.</td>
      <td>10890.0</td>
      <td>11.512925</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 63 columns</p>
</div>




```python
plot_distribution(training_data, label='Log Sale Price');
```


    
![png](cook-county-analysis_files/cook-county-analysis_26_0.png)
    



```python
training_data['Log Sale Price'].describe()
```




    count    168931.000000
    mean         12.168227
    std           0.999586
    min           6.214608
    25%          11.703546
    50%          12.278393
    75%          12.765688
    max          18.078190
    Name: Log Sale Price, dtype: float64




```python
training_data['Log Building Square Feet'] = np.log(training_data['Building Square Feet'])
```

I see a strong concentration of datapoints around the simple linear regression line. This gives us a good correlation.


```python
def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (DataFrame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than or equal to this will be removed
      upper (numeric): observations with values higher than this will be removed
    
    Output:
      a DataFrame with outliers removed
      
    Note: This function should not change mutate the contents of data.
    """  
    return data[(data[variable] <= upper) & (data[variable] >= lower)]
```


```python
pd.set_option('display.max_colwidth', None)
training_data.loc[0:5, 'Description']
```




    1                               This property, sold on 05/23/2018, is a one-story houeshold located at 2844 N LOWELL AVE.It has a total of 6 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.
    2                             This property, sold on 02/18/2016, is a one-story houeshold located at 11415 S PRAIRIE AVE.It has a total of 7 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.
    3    This property, sold on 07/23/2013, is a one-story with partially livable attics houeshold located at 2012 DOBSON ST.It has a total of 5 rooms, 3 of which are bedrooms, and 1.5 of which are bathrooms.
    4                                    This property, sold on 06/10/2016, is a one-story houeshold located at 104 SAUK TRL.It has a total of 5 rooms, 2 of which are bedrooms, and 1.0 of which are bathrooms.
    Name: Description, dtype: object




```python
q5a = [1, 2, 4, 6, 7, 8]
```


```python
def add_total_bedrooms(data):
    """
    Input:
      data (DataFrame): a DataFrame containing at least the Description column.

    Output:
      a Dataframe with a new column "Bedrooms" containing ints.

    """
    with_rooms = data.copy()
    #match number following a single space, "rooms," then a comma
    with_rooms['Bedrooms'] = with_rooms['Description'].str.findall(r' (\d+) rooms,').str[0].fillna(0).astype(int)
    return with_rooms

training_data = add_total_bedrooms(training_data)
```


```python
#checking that add_total_bedrooms worked
training_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PIN</th>
      <th>Property Class</th>
      <th>Neighborhood Code</th>
      <th>Land Square Feet</th>
      <th>Town Code</th>
      <th>Apartments</th>
      <th>Wall Material</th>
      <th>Roof Material</th>
      <th>Basement</th>
      <th>Basement Finish</th>
      <th>...</th>
      <th>Age Decade</th>
      <th>Pure Market Filter</th>
      <th>Garage Indicator</th>
      <th>Neigborhood Code (mapping)</th>
      <th>Town and Neighborhood</th>
      <th>Description</th>
      <th>Lot Size</th>
      <th>Log Sale Price</th>
      <th>Log Building Square Feet</th>
      <th>Bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13272240180000</td>
      <td>202</td>
      <td>120</td>
      <td>3780.0</td>
      <td>71</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>9.6</td>
      <td>1</td>
      <td>1.0</td>
      <td>120</td>
      <td>71120</td>
      <td>This property, sold on 05/23/2018, is a one-story houeshold located at 2844 N LOWELL AVE.It has a total of 6 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>3780.0</td>
      <td>12.560244</td>
      <td>6.904751</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25221150230000</td>
      <td>202</td>
      <td>210</td>
      <td>4375.0</td>
      <td>70</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>11.2</td>
      <td>1</td>
      <td>1.0</td>
      <td>210</td>
      <td>70210</td>
      <td>This property, sold on 02/18/2016, is a one-story houeshold located at 11415 S PRAIRIE AVE.It has a total of 7 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>4375.0</td>
      <td>9.998798</td>
      <td>6.810142</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10251130030000</td>
      <td>203</td>
      <td>220</td>
      <td>4375.0</td>
      <td>17</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>6.3</td>
      <td>1</td>
      <td>1.0</td>
      <td>220</td>
      <td>17220</td>
      <td>This property, sold on 07/23/2013, is a one-story with partially livable attics houeshold located at 2012 DOBSON ST.It has a total of 5 rooms, 3 of which are bedrooms, and 1.5 of which are bathrooms.</td>
      <td>4375.0</td>
      <td>12.323856</td>
      <td>7.068172</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 65 columns</p>
</div>




```python
sns.boxplot(data = training_data, x = 'Bedrooms', y = 'Log Sale Price')
plt.title('Distribution of Log Sale Price Depending on Number of Bedrooms')
```




    Text(0.5, 1.0, 'Distribution of Log Sale Price Depending on Number of Bedrooms')




    
![png](cook-county-analysis_files/cook-county-analysis_35_1.png)
    



```python
num_neighborhoods = (training_data['Neighborhood Code']).nunique()
num_neighborhoods
```




    193




```python
def plot_categorical(neighborhoods):
    fig, axs = plt.subplots(nrows=2)

    sns.boxplot(
        x='Neighborhood Code',
        y='Log Sale Price',
        data=neighborhoods,
        ax=axs[0],
    )

    sns.countplot(
        x='Neighborhood Code',
        data=neighborhoods,
        ax=axs[1],
    )

    # Draw median price
    axs[0].axhline(
        y=training_data['Log Sale Price'].median(), 
        color='red',
        linestyle='dotted'
    )

    # Label the bars with counts
    for patch in axs[1].patches:
        x = patch.get_bbox().get_points()[:, 0]
        y = patch.get_bbox().get_points()[1, 1]
        axs[1].annotate(f'{int(y)}', (x.mean(), y), ha='center', va='bottom')

    # Format x-axes
    axs[1].set_xticklabels(axs[1].xaxis.get_majorticklabels(), rotation=90)
    axs[0].xaxis.set_visible(False)

    # Narrow the gap between the plots
    plt.subplots_adjust(hspace=0.01)
    fig.suptitle("Neighborhoods in Cook County")
    plt.show()

plot_categorical(training_data)
```


    
![png](cook-county-analysis_files/cook-county-analysis_37_0.png)
    


Looks like I have run into the problem of overplotting again.

The graph is overplotted because **there are actually quite a few neighborhoods in this dataset**! For the clarity of our visualization, I will have to zoom in again on a few of them. The reason for this is that the visualization will become quite cluttered with a super dense x-axis.

Assign the variable `in_top_20_neighborhoods` to a copy of `training_data` that has been filtered to only contain rows corresponding to properties that are in one of the top 20 most populous neighborhoods. I define the “top 20 neighborhoods” as being the 20 neighborhood codes that have the greatest number of properties within them.


```python
(training_data['Neighborhood Code']).value_counts(20)
```




    Neighborhood Code
    30     0.051802
    80     0.042426
    10     0.037157
    70     0.034482
    50     0.032948
             ...   
    99     0.000030
    134    0.000030
    145    0.000018
    341    0.000006
    106    0.000006
    Name: proportion, Length: 193, dtype: float64




```python
top_20_neighborhood_codes = training_data['Neighborhood Code'].value_counts().head(20).index.tolist()
in_top_20_neighborhoods = training_data[training_data['Neighborhood Code'].isin(top_20_neighborhood_codes)]
```


```python
plot_categorical(neighborhoods=in_top_20_neighborhoods)
```


    
![png](cook-county-analysis_files/cook-county-analysis_41_0.png)
    



```python
def find_expensive_neighborhoods(data, n=3, metric=np.median):
    """
    Input:
      data (DataFrame): should contain at least a int-valued 'Neighborhood Code'
        and a numeric 'Log Sale Price' column
      n (int): the number of top values desired
      metric (function): function used for aggregating the data in each neighborhood.
        for example, np.median for median prices
    
    Output:
      a list of the the neighborhood codes of the top n highest-priced neighborhoods 
      as measured by the metric function
    """
    neighborhoods = data.groupby('Neighborhood Code').agg({'Log Sale Price': metric}).sort_values(by='Log Sale Price', ascending=False).index[:n]
    
    # This makes sure the final list contains the generic int type used in Python3, not specific ones used in numpy.
    return [int(code) for code in neighborhoods]

expensive_neighborhoods = find_expensive_neighborhoods(training_data, 3, np.median)
expensive_neighborhoods
```




    [44, 94, 93]




```python
def add_in_expensive_neighborhood(data, expensive_neighborhoods):
    """
    Input:
      data (DataFrame): a DataFrame containing a 'Neighborhood Code' column with values
        found in the codebook
      neighborhoods (list of strings): strings should be the names of neighborhoods
        pre-identified as expensive
    Output:
      DataFrame identical to the input with the addition of a binary
      in_expensive_neighborhood column
    """
    data['in_expensive_neighborhood'] = data['Neighborhood Code'].isin(expensive_neighborhoods).astype(int)
    return data

expensive_neighborhoods = find_expensive_neighborhoods(training_data, 3, np.median)
training_data = add_in_expensive_neighborhood(training_data, expensive_neighborhoods)
```


```python
def substitute_roof_material(data):
    """
    Input:
      data (DataFrame): a DataFrame containing a 'Roof Material' column.  Its values
                         should be limited to those found in the codebook
    Output:
      DataFrame identical to the input except with a refactored 'Roof Material' column
    """
    data = data.replace({'Roof Material':{
        1: 'Shingle/Asphalt',
        2: 'Tar & Gravel',
        3: 'Slate',
        4: 'Shake',
        5: 'Tile',
        6: 'Other'
    }})
    return data
    
training_data_mapped = substitute_roof_material(training_data)
training_data_mapped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PIN</th>
      <th>Property Class</th>
      <th>Neighborhood Code</th>
      <th>Land Square Feet</th>
      <th>Town Code</th>
      <th>Apartments</th>
      <th>Wall Material</th>
      <th>Roof Material</th>
      <th>Basement</th>
      <th>Basement Finish</th>
      <th>...</th>
      <th>Pure Market Filter</th>
      <th>Garage Indicator</th>
      <th>Neigborhood Code (mapping)</th>
      <th>Town and Neighborhood</th>
      <th>Description</th>
      <th>Lot Size</th>
      <th>Log Sale Price</th>
      <th>Log Building Square Feet</th>
      <th>Bedrooms</th>
      <th>in_expensive_neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13272240180000</td>
      <td>202</td>
      <td>120</td>
      <td>3780.0</td>
      <td>71</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Shingle/Asphalt</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1.0</td>
      <td>120</td>
      <td>71120</td>
      <td>This property, sold on 05/23/2018, is a one-story houeshold located at 2844 N LOWELL AVE.It has a total of 6 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>3780.0</td>
      <td>12.560244</td>
      <td>6.904751</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25221150230000</td>
      <td>202</td>
      <td>210</td>
      <td>4375.0</td>
      <td>70</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Shingle/Asphalt</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>1.0</td>
      <td>210</td>
      <td>70210</td>
      <td>This property, sold on 02/18/2016, is a one-story houeshold located at 11415 S PRAIRIE AVE.It has a total of 7 rooms, 3 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>4375.0</td>
      <td>9.998798</td>
      <td>6.810142</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10251130030000</td>
      <td>203</td>
      <td>220</td>
      <td>4375.0</td>
      <td>17</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Shingle/Asphalt</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>1.0</td>
      <td>220</td>
      <td>17220</td>
      <td>This property, sold on 07/23/2013, is a one-story with partially livable attics houeshold located at 2012 DOBSON ST.It has a total of 5 rooms, 3 of which are bedrooms, and 1.5 of which are bathrooms.</td>
      <td>4375.0</td>
      <td>12.323856</td>
      <td>7.068172</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31361040550000</td>
      <td>202</td>
      <td>120</td>
      <td>8400.0</td>
      <td>32</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Shingle/Asphalt</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>1.0</td>
      <td>120</td>
      <td>32120</td>
      <td>This property, sold on 06/10/2016, is a one-story houeshold located at 104 SAUK TRL.It has a total of 5 rooms, 2 of which are bedrooms, and 1.0 of which are bathrooms.</td>
      <td>8400.0</td>
      <td>10.025705</td>
      <td>6.855409</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30314240080000</td>
      <td>203</td>
      <td>181</td>
      <td>10890.0</td>
      <td>37</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Shingle/Asphalt</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>1.0</td>
      <td>181</td>
      <td>37181</td>
      <td>This property, sold on 10/26/2017, is a one-story with partially livable attics houeshold located at 2820 186TH ST.It has a total of 6 rooms, 4 of which are bedrooms, and 1.5 of which are bathrooms.</td>
      <td>10890.0</td>
      <td>11.512925</td>
      <td>7.458186</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>




```python
from sklearn.preprocessing import OneHotEncoder

def ohe_roof_material(data):
    """
    One-hot-encodes roof material. New columns are of the form "Roof Material_MATERIAL".
    """
    enc = OneHotEncoder()
    enc.fit(data[['Roof Material']])
    new_columns = pd.DataFrame(enc.transform(data[['Roof Material']]).toarray(),
                               columns = enc.get_feature_names_out(),
                               index = data.index)
    return pd.concat((data, new_columns), axis = 1)

training_data_ohe = ohe_roof_material(training_data_mapped)
# This line of code will display only the one-hot-encoded columns in training_data_ohe that 
# have names that begin with “Roof Material_" 
training_data_ohe.filter(regex='^Roof Material_').head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Roof Material_Other</th>
      <th>Roof Material_Shake</th>
      <th>Roof Material_Shingle/Asphalt</th>
      <th>Roof Material_Slate</th>
      <th>Roof Material_Tar &amp; Gravel</th>
      <th>Roof Material_Tile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<hr style="border: 1px solid #fdb515;" />
<hr style="border: 5px solid #003262;" />
<br/><br/>

This dataset is split into a training/validation set and a testing set. Importantly, the test set does not contain values for my target variable, `Sale Price`. In this project, I train a model on the training/validation set then use this model to predict the `Sale Price`s of the test set. In the cell below, I load the training/validation set into the `DataFrame` `training_val_data` and the test set into the `DataFrame` `test_data`.


```python
training_val_data = pd.read_csv("data/cook_county_train_val.csv", index_col='Unnamed: 0')
test_data = pd.read_csv("data/cook_county_contest_test.csv", index_col='Unnamed: 0')
```

As a good sanity check, I verify that the data shape matches the description.


```python
# 204792 observations and 62 features in training data
assert training_val_data.shape == (204792, 62)
# 55311 observations and 61 features in test data
assert test_data.shape == (55311, 61)
# Sale Price is provided in the training/validation data
assert 'Sale Price' in training_val_data.columns.values
# Sale Price is hidden in the test data
assert 'Sale Price' not in test_data.columns.values
```


```python
training_val_data.columns.values
```




    array(['PIN', 'Property Class', 'Neighborhood Code', 'Land Square Feet',
           'Town Code', 'Apartments', 'Wall Material', 'Roof Material',
           'Basement', 'Basement Finish', 'Central Heating', 'Other Heating',
           'Central Air', 'Fireplaces', 'Attic Type', 'Attic Finish',
           'Design Plan', 'Cathedral Ceiling', 'Construction Quality',
           'Site Desirability', 'Garage 1 Size', 'Garage 1 Material',
           'Garage 1 Attachment', 'Garage 1 Area', 'Garage 2 Size',
           'Garage 2 Material', 'Garage 2 Attachment', 'Garage 2 Area',
           'Porch', 'Other Improvements', 'Building Square Feet',
           'Repair Condition', 'Multi Code', 'Number of Commercial Units',
           'Estimate (Land)', 'Estimate (Building)', 'Deed No.', 'Sale Price',
           'Longitude', 'Latitude', 'Census Tract',
           'Multi Property Indicator', 'Modeling Group', 'Age', 'Use',
           "O'Hare Noise", 'Floodplain', 'Road Proximity', 'Sale Year',
           'Sale Quarter', 'Sale Half-Year', 'Sale Quarter of Year',
           'Sale Month of Year', 'Sale Half of Year', 'Most Recent Sale',
           'Age Decade', 'Pure Market Filter', 'Garage Indicator',
           'Neigborhood Code (mapping)', 'Town and Neighborhood',
           'Description', 'Lot Size'], dtype=object)



1. Buyers need to know housing prices to gauge how to manage their purchasing budget and timing.
2. Sellers would likely prefer to sell when the housing prices are high rather than low. Knowing housing prices would be beneficial for them to make the most profit on their property.
3. Real estate agents would be interested in knowing housing prices as Ill since it directly impacts their commission income.

I find that an assessment process which systematically overvalues lower-priced properties and undervalues higher-priced ones creates significant barriers for low-income buyers. If an affordable home is assessed too high, it can mislead budget-constrained buyers into thinking the property is out of reach, which effectively prices them out. Conversely, buyers with higher budgets might benefit from under-assessed high-value homes, gaining access to properties they otherwise might have overlooked. This dynamic feels especially unfair because it disproportionately disadvantages those with fewer housing options to begin with.

The assessment process failed to meet basic standards of accuracy and fairness. Due to these skewed valuations, property taxes ended up being higher for low-income homeowners and lower for high-income ones, contributing to an inequitable gap between Black and white property owners. While an appeals process technically exists, it is not equally accessible in practice. Some communities rarely file appeals, often because of limited resources or awareness, while others appeal frequently. Notably, the areas that almost never appeal are also the ones where properties are consistently overvalued, further entrenching the imbalance.

Encoding race also made it possible for the real estate system to carry out racial discrimination widely. Non-white property owners continue to carry the burden of a long history of segregation, redlining, and systemic exclusion. The appeals process, while technically available to all, implicitly favors white property owners who are more likely to have the financial resources and legal knowledge to navigate it. As a result, overvalued assessments are more likely to remain unchallenged for non-white homeowners, making these inflated property values and their associated tax burdens more entrenched and difficult to correct.


```python
# This makes the train-validation split in this section reproducible across different runs 
# of the notebook. You do not need this line to run train_val_split in general

# DO NOT CHANGE THIS LINE
np.random.seed(1337)
# DO NOT CHANGE THIS LINE

def train_val_split(data):
    """ 
    Takes in a DataFrame `data` and randomly splits it into two smaller DataFrames 
    named `train` and `validation` with 80% and 20% of the data, respectively. 
    """
    
    data_len = data.shape[0]
    shuffled_indices = np.random.permutation(data_len)
    split_indices = int(data_len * 0.8)
    train = data.iloc[shuffled_indices[:split_indices]]
    validation = data.iloc[shuffled_indices[split_indices:]]
   
    return train, validation
train, validation = train_val_split(training_val_data)
```

---

## 4. Data Processing Pipeline & Modeling

To streamline my workflow, I created a pipeline function that encapsulates all the necessary feature engineering steps for the first model. Rather than calling each transformation function manually, the pipeline wraps them into a single reusable function that takes in the raw dataset and returns X and Y.

This approach ensures consistency across training and validation splits, and makes the notebook cleaner and easier to maintain.

The helper functions used in this pipeline, such as remove_outliers, log_transform, and add_total_bedrooms, are defined earlier in the notebook.

*Model training: feature-target split, model definition, and fitting.*

### Baseline Model (Model 1)


```python
from feature_func_inline import *    # Import functions from Project A1


def process_data_simple(data):
    # Remove outliers
    data = remove_outliers(data, 'Sale Price', lower=499)
    # Create Log Sale Price column
    data = log_transform(data, 'Sale Price')
    # Create Bedroom column
    data = add_total_bedrooms(data)
    # Select X and Y from the full data
    X = data[['Bedrooms']]
    Y = data['Log Sale Price']
    return X, Y

# Reload the data
full_data = pd.read_csv("data/cook_county_train.csv")

# Process the data using the pipeline for the first model.
np.random.seed(1337)
train_m1, valid_m1 = train_val_split(full_data)
X_train_m1_simple, Y_train_m1_simple = process_data_simple(train_m1)
X_valid_m1_simple, Y_valid_m1_simple = process_data_simple(valid_m1)

# Take a look at the result
display(X_train_m1_simple.head())
display(Y_train_m1_simple.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163803</th>
      <td>4</td>
    </tr>
    <tr>
      <th>69817</th>
      <td>8</td>
    </tr>
    <tr>
      <th>11374</th>
      <td>5</td>
    </tr>
    <tr>
      <th>140562</th>
      <td>5</td>
    </tr>
    <tr>
      <th>37232</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



    163803    12.608199
    69817     12.409013
    11374     11.630709
    140562     8.517193
    37232     11.641758
    Name: Log Sale Price, dtype: float64


### `.pipe`

Doing so is a bit convoluted. Alternatively, I build a pipeline using `pd.DataFrame.pipe` ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html)).

The following function `process_data_pipe` takes in a `DataFrame` `data`, a list `pipeline_functions` containing 3-element tuples `(function, arguments, keyword_arguments)` that will be called on `data` in the pipeline, and the label `prediction_col` that represents the column of our target variable (`Sale Price` in this case). I can use this function with each of the tuples passed in through `pipeline_functions`.


```python
# Define process_data_pipe.
def process_data_pipe(data, pipeline_functions, prediction_col):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    X = data.drop(columns=[prediction_col])
    Y = data.loc[:, prediction_col]
    return X, Y
```


```python
# Reload the data
full_data = pd.read_csv("data/cook_county_train.csv")

# Process the data using the pipeline for the first model
np.random.seed(1337)
train_m1, valid_m1 = train_val_split(full_data)

# Helper function
def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

# Pipelines, a list of tuples
m1_pipelines = [
    (remove_outliers, None, {
        'column': 'Sale Price',
        'lower': 499,
    }),
    (log_transform, None, {'column': 'Sale Price'}),
    (add_total_bedrooms, None, None),
    (select_columns, ['Log Sale Price', 'Bedrooms'], None)
]

X_train_m1, Y_train_m1 = process_data_pipe(train_m1, m1_pipelines, 'Log Sale Price')
X_valid_m1, Y_valid_m1 = process_data_pipe(valid_m1, m1_pipelines, 'Log Sale Price')

# Take a look at the result
# It should be the same above as the result returned by process_data_simple
display(X_train_m1.head())
display(Y_train_m1.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163803</th>
      <td>4</td>
    </tr>
    <tr>
      <th>69817</th>
      <td>8</td>
    </tr>
    <tr>
      <th>11374</th>
      <td>5</td>
    </tr>
    <tr>
      <th>140562</th>
      <td>5</td>
    </tr>
    <tr>
      <th>37232</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



    163803    12.608199
    69817     12.409013
    11374     11.630709
    140562     8.517193
    37232     11.641758
    Name: Log Sale Price, dtype: float64


### Intermediate Model (Model 2)
This model uses more features and a more complex pipeline to predict sale prices.
It includes additional transformations and feature engineering steps.


```python
# Set the random seed for reproducibility
np.random.seed(1337)

# Split the full dataset into training and validation sets for Model 2
train_m2, valid_m2 = train_val_split(full_data)

# Define the processing pipeline for Model 2
m2_pipelines = [
    (remove_outliers, None, {
        'column': 'Sale Price',  # fixed key name
        'lower': 499,
    }),
    (log_transform, None, {'column': 'Sale Price'}),
    (log_transform, None, {'column': 'Building Square Feet'}),
    (add_total_bedrooms, None, None),
    (select_columns, ['Log Sale Price', 'Bedrooms', 'Log Building Square Feet'], None)
]

# Apply the pipeline to training and validation data
X_train_m2, Y_train_m2 = process_data_pipe(train_m2, m2_pipelines, 'Log Sale Price')
X_valid_m2, Y_valid_m2 = process_data_pipe(valid_m2, m2_pipelines, 'Log Sale Price')

# Preview results
display(X_train_m2.head())
display(Y_train_m2.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bedrooms</th>
      <th>Log Building Square Feet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163803</th>
      <td>4</td>
      <td>6.860664</td>
    </tr>
    <tr>
      <th>69817</th>
      <td>8</td>
      <td>7.397562</td>
    </tr>
    <tr>
      <th>11374</th>
      <td>5</td>
      <td>7.051856</td>
    </tr>
    <tr>
      <th>140562</th>
      <td>5</td>
      <td>6.813445</td>
    </tr>
    <tr>
      <th>37232</th>
      <td>7</td>
      <td>7.031741</td>
    </tr>
  </tbody>
</table>
</div>



    163803    12.608199
    69817     12.409013
    11374     11.630709
    140562     8.517193
    37232     11.641758
    Name: Log Sale Price, dtype: float64


I first initialize a [`sklearn.linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) object for both of our models. I set the `fit_intercept = True` to ensure that the linear model has a non-zero intercept (i.e., a bias term).


```python
linear_model_m1 = lm(fit_intercept=True)
linear_model_m2 = lm(fit_intercept=True)
```


```python
X_train_m1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163803</th>
      <td>4</td>
    </tr>
    <tr>
      <th>69817</th>
      <td>8</td>
    </tr>
    <tr>
      <th>11374</th>
      <td>5</td>
    </tr>
    <tr>
      <th>140562</th>
      <td>5</td>
    </tr>
    <tr>
      <th>37232</th>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>71538</th>
      <td>8</td>
    </tr>
    <tr>
      <th>153946</th>
      <td>6</td>
    </tr>
    <tr>
      <th>117415</th>
      <td>7</td>
    </tr>
    <tr>
      <th>9448</th>
      <td>8</td>
    </tr>
    <tr>
      <th>188605</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>135116 rows × 1 columns</p>
</div>



---
## 5. Evaluation & Metrics

*Performance metrics: RMSE_dollars, bias analysis, subgroup evaluation.*


```python
# Fit the 1st model
linear_model_m1.fit(X_train_m1, Y_train_m1)
# Compute the fitted and predicted values of Log Sale Price for the 1st model
Y_fitted_m1 = linear_model_m1.predict(X_train_m1)
Y_predicted_m1 = linear_model_m1.predict(X_valid_m1)

# Fit the 2nd model
linear_model_m2.fit(X_train_m2, Y_train_m2)
# Compute the fitted and predicted values of Log Sale Price for the 2nd model
Y_fitted_m2 = linear_model_m2.predict(X_train_m2)
Y_predicted_m2 = linear_model_m2.predict(X_valid_m2)
```


```python
def rmse(predicted, actual):
    """
    Calculates RMSE from actual and predicted values.
    Input:
      predicted (1D array): Vector of predicted/fitted values
      actual (1D array): Vector of actual values
    Output:
      A float, the RMSE value.
    """
    return np.sqrt(np.mean((actual - predicted)**2))
```


```python
plt.scatter(x = Y_valid_m2, y = Y_valid_m2 - Y_predicted_m2, s = 0.1, alpha = 0.5)
plt.xlabel('Original Log Sale Price')
plt.ylabel('Residual')
plt.title('Residual vs Original Log Sale Price')
```




    Text(0.5, 1.0, 'Residual vs Original Log Sale Price')




    
![png](cook-county-analysis_files/cook-county-analysis_72_1.png)
    


While my model explains some of the variability in price, there is certainly still a lot of room for improvement to be made. One reason is I have been only utilizing 1 or 2 features (out of a total of 70+) so far!


```python
# Define model evaluation model using RMSE
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X, y, label=""):
    """
    Fits the model on X and y, prints and returns RMSE.

    Parameters:
        model: An sklearn estimator (e.g., LinearRegression)
        X: Features DataFrame
        y: Target Series
        label (str): Optional label to identify the run

    Returns:
        float: RMSE score on the training data
    """
    model.fit(X, y)
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    if label:
        print(f"{label} RMSE: {rmse:.4f}")
    else:
        print(f"RMSE: {rmse:.4f}")
    return rmse
```


```python
rnse_1 = evaluate_model(linear_model_m1, X_train_m1, Y_train_m1, "Model 1")
rnse_2 = evaluate_model(linear_model_m2, X_train_m2, Y_train_m2, "Model 2")
```

    Model 1 RMSE: 0.8674
    Model 2 RMSE: 0.8059
    

Model 1's RMSE at 0.8674 means that accounting for bedrooms only captures little of the true price variation, leaving lots of features unmodeled. Model 2's RMSE at 0.8059 show a meaningful drop in error. Model 2 introduces building size along with the bedroom count, which explains more of the variation but lacks persuasion. To improve models, I need to add a full suite of engineered and transformed features to drive RMSE down even further.

### Final Model (Model 3)
Features (13 total): log-transformed sale price, building/land sq ft, assessor estimates; engineered fireplaces, repair condition; binary central air; one-hot room counts (1–6+); Estimator: LinearRegression; Evaluation: RMSE.


```python
from sklearn.linear_model import LinearRegression

def remove_outliers(data, column, lower=-np.inf, upper=np.inf):
    return data[(data[column] <= upper) & (data[column] >= lower)]

def rm_outliers_pct(data, columns, l=25, h=75):
    for col in columns:
        low = np.percentile(data[col], l)
        high = np.percentile(data[col], h)
        data = remove_outliers(data, col, lower=low, upper=high)
    return data

def log_cols(data, columns):
    for col in columns:
        data['Log ' + col] = np.log(data[col].clip(lower=1e-9))  # avoid log(0)
    return data

def exp_cols(data, columns, n):
    for col in columns:
        data['Exp ' + col] = data[col] ** n
    return data

def process_data_final(data, is_test_set=False):
    data = log_cols(data, ['Building Square Feet', 'Land Square Feet', 'Estimate (Building)', 'Estimate (Land)'])
    data = exp_cols(data, ['Fireplaces', 'Repair Condition'], n=3)

    data = add_total_bedrooms(data)
    data = pd.get_dummies(data, prefix='RM', columns=['Roof Material'])
    
    if not is_test_set:
        data['Log Sale Price'] = np.log(data['Sale Price'].clip(lower=1e-9))
        data = rm_outliers_pct(data, ['Sale Price', 'Land Square Feet', 'Estimate (Building)', 'Estimate (Land)'])
        keep_cols = ['Log Sale Price', 'Log Building Square Feet', 'Log Land Square Feet',
                     'Log Estimate (Building)', 'Log Estimate (Land)', 'Exp Fireplaces', 'Exp Repair Condition',
                     'Central Air', 'Bedrooms', 'RM_1.0', 'RM_2.0', 'RM_3.0', 'RM_4.0', 'RM_5.0', 'RM_6.0']
        data = data[keep_cols]
        X = data.drop(['Log Sale Price'], axis=1)
        y = data['Log Sale Price']
        return X, y
    else:
        else_cols = ['Log Building Square Feet', 'Log Land Square Feet', 'Log Estimate (Building)', 'Log Estimate (Land)',
                     'Exp Fireplaces', 'Exp Repair Condition', 'Central Air', 'Bedrooms', 'RM_1.0', 'RM_2.0', 'RM_3.0',
                     'RM_4.0', 'RM_5.0', 'RM_6.0']
        data = data[else_cols]
        return data

# Load and evaluate
df = pd.read_csv("data/cook_county_train.csv")
X_final, y_final = process_data_final(df)

model = LinearRegression()
evaluate_model(model, X_final, y_final, label="Final Model")
```

    Final Model RMSE: 0.3716
    




    0.37161837157363287




```python
from datetime import datetime
from IPython.display import display, HTML

# Load test set
df_test = pd.read_csv("data/cook_county_contest_test.csv")

# Preprocess the test data
X_test = process_data_final(df_test, is_test_set=True)

# Clean inf/NaN in test features
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Fit model on full training data
df_train = pd.read_csv("data/cook_county_train.csv")
X_train, y_train = process_data_final(df_train)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict on cleaned test data
y_test_pred = model.predict(X_test)

# Build submission DataFrame
submission_df = pd.DataFrame({
    "Id": df_test['Unnamed: 0'],
    "Value": y_test_pred
})

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submission_{timestamp}.csv"
submission_df.to_csv(filename, index=False)

# Show download link
display(HTML(f"✅ Download test prediction <a href='{filename}' download>here</a>."))
```


✅ Download test prediction <a href='submission_20250807_151903.csv' download>here</a>.



```python
# Check if prediction is reasonable. 
submission_df["Value"].describe()
```




    count    55311.000000
    mean        12.166851
    std          1.131620
    min         -1.489455
    25%         11.728178
    50%         12.203682
    75%         12.660360
    max         16.757047
    Name: Value, dtype: float64



These prediction results show that my model is capturing the middle 90 % of Cook County sales quite well (median ≈ $200 k).


```python
# Run the cell below
train_df = pd.read_csv('data/cook_county_train.csv')
X, Y = process_data_final(train_df)
model = lm(fit_intercept=True)
model.fit(X, Y)
Y_pred = model.predict(X)
```


```python
# Obtain the two subsets of data.
small_interval = (8, 11)
big_interval = (11, 14)
Y = pd.DataFrame(Y)
Y_small = Y[(Y['Log Sale Price'] > small_interval[0]) & (Y['Log Sale Price'] < small_interval[1])]['Log Sale Price']
Y_big = Y[(Y['Log Sale Price'] > big_interval[0]) & (Y['Log Sale Price'] < big_interval[1])]['Log Sale Price']
X_small = X.loc[Y_small.index]
X_big = X.loc[Y_big.index]
```

**Compute the RMSE of the model's predictions on each subset separately** and assign those values to `rmse_small` and `rmse_big` respectively. Here, I am looking for the RMSE **with regard to `Sale Price`**. I have to exponentiate my predictions and response vectors before computing the RMSE using the `rmse` function defined earlier!

Separately, I also want to understand whether the proportion of houses in each interval that the model overestimates the value of the actual `Sale Price`. To that end, **compute the proportion of predictions strictly greater than the corresponding true price in each subset** and assign it to `prop_overest_small` and `prop_overest_big` respectively. For example, if I am working with a dataset of 3 houses wherein the actual `Log Sale Price`s Ire [10, 11, 12] and the model predictions Ire [5, 15, 13], then the proportion of houses with overestimated values would be 2/3.  


```python
rmse_small = rmse(np.exp(model.predict(X_small)), np.exp(Y_small))
rmse_big = rmse(np.exp(model.predict(X_big)), np.exp(Y_big))

prop_overest_small = np.mean(np.exp(model.predict(X_small)) >= np.exp(Y_small))
prop_overest_big = np.mean(np.exp(model.predict(X_big)) >= np.exp(Y_big))

print(f"The RMSE for properties with log sale prices in the interval {small_interval} is {np.round(rmse_small)}")
print(f"The RMSE for properties with log sale prices in the interval {big_interval} is {np.round(rmse_big)}")
print(f"The percentage of overestimated values for properties with log sale prices in the interval {small_interval} is {100 * np.round(prop_overest_small, 3)}%")
print(f"The percentage of overestimated values for properties with log sale prices in the interval {big_interval} is {100 * np.round(prop_overest_big, 3)}%")
```

    The RMSE for properties with log sale prices in the interval (8, 11) is 82749.0
    The RMSE for properties with log sale prices in the interval (11, 14) is 53394.0
    The percentage of overestimated values for properties with log sale prices in the interval (8, 11) is 100.0%
    The percentage of overestimated values for properties with log sale prices in the interval (11, 14) is 40.9%
    


```python
def rmse_interval(X, Y, start, end):
    '''
    Given a design matrix X and response vector Y, computes the RMSE for a subset of values 
    wherein the corresponding Log Sale Price lies in the interval (start, end).

    Input: 
    X - 2D DataFrame representing the design matrix.
    Y - 1D DataFrame consisting of a single column labeled 'Log Sale Price'.
    start - A float specifying the start of the interval (exclusive).
    end - A float specifying the end of the interval (exclusive).
    '''
    
    Y_subset = Y[(Y['Log Sale Price'] > start) & (Y['Log Sale Price'] < end)]['Log Sale Price']
    X_subset = X.loc[Y_subset.index]

    if len(X_subset) == 0:
        return 0

    Y_pred_subset = model.predict(X_subset)
    Y_true_subset = np.exp(Y_subset)
    Y_pred_subset = np.exp(Y_pred_subset)
        
    rmse_subset = rmse(Y_true_subset, Y_pred_subset)
    return rmse_subset
    
def prop_overest_interval(X, Y, start, end):
    '''
    Given a design matrix X and response vector Y, computes prop_overest for a subset of values 
    wherein the corresponding Log Sale Price lies in the interval (start, end).

    Input: 
    X - 2D DataFrame representing the design matrix.
    Y - 1D DataFrame consisting of a single column labeled 'Log Sale Price'.
    start - A float specifying the start of the interval (exclusive).
    end - A float specifying the end of the interval (exclusive).
    '''
    
    Y_subset = Y[(Y['Log Sale Price'] > start) & (Y['Log Sale Price'] < end)]['Log Sale Price']
    X_subset = X.loc[Y_subset.index]

    # DO NOT MODIFY THESE TWO LINES
    if len(X_subset) == 0:
        return 0

    Y_pred_subset = model.predict(X_subset)
    Y_true_subset = np.exp(Y_subset)
    Y_pred_subset = np.exp(Y_pred_subset)

    prop_subset = np.mean(Y_pred_subset > Y_true_subset)
    return prop_subset
```


```python
# Generate the plot for RMSE over different intervals of Log Sale Price
rmses = []
for i in np.arange(8, 14, 0.5):
    rmses.append(rmse_interval(X, Y, i, i + 0.5))
plt.figure(figsize = (7, 7))
plt.bar(x = np.arange(8.25, 14.25, 0.5), height = rmses, edgecolor = 'black', width = 0.5)
plt.title('RMSE over different intervals of Log Sale Price')
plt.xlabel('Log Sale Price')
plt.ylabel('RMSE');
```


    
![png](cook-county-analysis_files/cook-county-analysis_87_0.png)
    



```python
# Generate the plot for percentage of overestimated values over different intervals of Log Sale Price
props = []
for i in np.arange(8, 14, 0.5):
    props.append(prop_overest_interval(X, Y, i, i + 0.5) * 100)
plt.figure(figsize = (7, 7))
plt.bar(x = np.arange(8.25, 14.25, 0.5), height = props, edgecolor = 'black', width = 0.5)
plt.title('Percentage of House Values Overestimated over different intervals of Log Sale Price')
plt.xlabel('Log Sale Price')
plt.ylabel('Percentage of House Values that were Overestimated (%)');
```


    
![png](cook-county-analysis_files/cook-county-analysis_88_0.png)
    


The second plot titled *'Percentage of House Values Overestimated over different intervals of Log Sale Price'* shows a key feature: downward‐sloping curve of overestimation rates as you move from low‐value to high‐value home.

At the lowest log sale prices (left side), nearly 100 % of homes are overvalued.

As you move right toward higher‐value properties, that percentage steadily falls, reaching much lower levels at the top end.

Because overvaluation (and thus higher tax burden) is concentrated among the lower‐value homes, this matches the definition of *regressive taxation* (i.e. the “tax” impact is heavier on the less expensive properties). In other words, the plot shows that poorer homeowners face proportionally larger overassessments than wealthier homeowners.

---

## 6. Conclusions & Next Steps

*Key findings, limitations, and potential extensions.*

**Key Findings**  
- **Strong Predictive Gains**: Progressing from a bedrooms-only model (RMSE = 0.8674) to a full 13-feature specification (RMSE = 0.3716) demonstrates the impact of targeted feature engineering—log transforms of size and assessor estimates, nonlinear expansions of condition variables, and categorical one-hot encoding.  
- **Regressive Assessment Bias**: Residual analysis shows lower-priced homes are systematically overvalued while higher-priced homes are slightly undervalued, producing a regressive taxation pattern that disproportionately burdens low-income homeowners.  
- **Equity Implications**: Assessment errors intersect with neighborhood demographics, tax-appeal disparities, and historical segregation, illustrating how an otherwise accurate model can perpetuate systemic unfairness.

**Limitations**  
1. **Omitted Spatial Variables**  
   - Excluding explicit geographic features (e.g., neighborhood codes, GIS coordinates) leaves spatially correlated market dynamics unmodeled.  
2. **Linear Specification Constraints**  
   - Purely linear regression cannot capture complex interactions (e.g., how square footage impacts value differently by neighborhood) beyond basic polynomial expansions.  
3. **Outlier Treatment**  
   - Trimming to the 25th–75th percentile range increases robustness but may discard legitimate transactions at market extremes, biasing insights on the most and least expensive homes.  
4. **Data Quality & Coverage**  
   - Reliance on assessor-reported features risks incorporating reporting errors or omissions, especially for interior conditions or recent renovations.

**Potential Extensions**  
1. **Spatial & Hierarchical Modeling**  
   - Incorporate geospatial predictors or use mixed-effects models to explicitly account for neighborhood-level random effects and reduce spatial bias.  
2. **Fairness-Aware Algorithms**  
   - Explore constrained optimization or post-processing methods that penalize disparate error rates across value strata, ensuring equity objectives alongside accuracy.  
3. **Nonlinear & Ensemble Techniques**  
   - Test tree-based models (Random Forests, Gradient Boosting) or kernel methods to capture complex interactions, while leveraging SHAP or partial-dependence plots for interpretability.  
4. **Advanced Outlier & Missing-Data Strategies**  
   - Replace simple percentile trimming with robust techniques (e.g., isolation forests) and apply multiple-imputation for missing or zero-value inputs in both train and test sets.  
5. **Interactive Dashboard & Stakeholder Engagement**  
   - Develop a dashboard enabling policymakers and community advocates to visualize residual distributions by neighborhood or demographic group, fostering transparency and targeted appeal reforms.  
6. **Longitudinal & Policy Impact Analysis**  
   - Extend the dataset across multiple years to track how assessment errors evolve, and link residuals to actual appeal outcomes or tax-delinquency rates to quantify real-world impacts.



