<a href="https://colab.research.google.com/github/Gnommer/Automated-Ticketing-System/blob/preprocessing/colab_version.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#### Cloning Github Repo


```
import os
import shutil
from getpass import getpass

username = input('Github username: ')
password = getpass('Github password: ')
os.environ['GITHUB_AUTH'] = username + ':' + password

!git clone https://$GITHUB_AUTH@github.com/Gnommer/Automated-Ticketing-System --branch preprocessing
!pip3 install -r requirements.txt
!wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
!pip3 -m spacy download en_core_web_sm
os.chdir("Automated-Ticketing-System")
import warnings
warnings.filterwarnings("ignore")
```

#### Importing Dependencies


```
import pandas as pd
from Pipelines import NLP_Pipeline

import matplotlib.pyplot as plt
```


```
pipeline1 = NLP_Pipeline(file_path="Dataset/Synthetic_Dataset.xlsx")
```

#### Loading Dataset


```
df = pipeline1.read_dataset()
```


```
df.head()
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
      <th>Short description</th>
      <th>Description</th>
      <th>Caller</th>
      <th>Assignment group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>login issue</td>
      <td>-verified user details.(employee# &amp; manager na...</td>
      <td>spxjnwir pjlcoqds</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>outlook</td>
      <td>\r\n\r\nreceived from: hmjdrvpb.komuaywn@gmail...</td>
      <td>hmjdrvpb komuaywn</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cant log in to vpn</td>
      <td>\r\n\r\nreceived from: eylqgodm.ybqkwiam@gmail...</td>
      <td>eylqgodm ybqkwiam</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>unable to access hr_tool page</td>
      <td>unable to access hr_tool page</td>
      <td>xbkucsvz gcpydteq</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>skype error</td>
      <td>skype error</td>
      <td>owlgqjme qhcozdfx</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Exploratory Data Analysis

##### Shape of the Data Frame


```
df.shape
```




    (8500, 4)



##### Missing values


```
pd.DataFrame(df.isna().sum()).T
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
      <th>Short description</th>
      <th>Description</th>
      <th>Caller</th>
      <th>Assignment group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



there are 8 records with short description which is missing
and 1 record with the description missing


```
df[df.isna()["Short description"]]
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
      <th>Short description</th>
      <th>Description</th>
      <th>Caller</th>
      <th>Assignment group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2604</th>
      <td>NaN</td>
      <td>\r\n\r\nreceived from: ohdrnswl.rezuibdt@gmail...</td>
      <td>ohdrnswl rezuibdt</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3383</th>
      <td>NaN</td>
      <td>\r\n-connected to the user system using teamvi...</td>
      <td>qftpazns fxpnytmk</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3906</th>
      <td>NaN</td>
      <td>-user unable  tologin to vpn.\r\n-connected to...</td>
      <td>awpcmsey ctdiuqwe</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3910</th>
      <td>NaN</td>
      <td>-user unable  tologin to vpn.\r\n-connected to...</td>
      <td>rhwsmefo tvphyura</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3915</th>
      <td>NaN</td>
      <td>-user unable  tologin to vpn.\r\n-connected to...</td>
      <td>hxripljo efzounig</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3921</th>
      <td>NaN</td>
      <td>-user unable  tologin to vpn.\r\n-connected to...</td>
      <td>cziadygo veiosxby</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3924</th>
      <td>NaN</td>
      <td>name:wvqgbdhm fwchqjor\nlanguage:\nbrowser:mic...</td>
      <td>wvqgbdhm fwchqjor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4341</th>
      <td>NaN</td>
      <td>\r\n\r\nreceived from: eqmuniov.ehxkcbgj@gmail...</td>
      <td>eqmuniov ehxkcbgj</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```
df[df.isna()["Description"]]
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
      <th>Short description</th>
      <th>Description</th>
      <th>Caller</th>
      <th>Assignment group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4395</th>
      <td>i am locked out of skype</td>
      <td>NaN</td>
      <td>viyglzfo ajtfzpkb</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



so we will be removing these records for now. the system should prevent the cases from accepting tickets without descriptions. or we have to generate additional features to supplement this problem


```
df = df.dropna()
```

#### Distributions of the labels

There are more Group 0 tickets than others.


```
plt.figure(figsize=(20, 5))
df["Assignment group"].value_counts()[:10].plot(kind='bar')
plt.yscale('linear')
plt.ylabel("Count of Tickets")
plt.xlabel("Groups")
plt.show()
```


![png](output_20_0.png)


There are clear bins formed for the complaints when the issues are explored in a log scale. Group 0 is the highest


```
plt.figure(figsize=(20, 5))
df["Assignment group"].value_counts().plot(kind='bar')
plt.yscale('log')
plt.ylabel("Count of Tickets")
plt.xlabel("Groups")
plt.show()
```


![png](output_22_0.png)


#### Presence of different languages

As we can see below there are tickets which are not completely in english. this ticket is in German. so we need to write a strategy to find the languages of the tickets based on the description and short description.


```
df[df.index == 255]
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
      <th>Short description</th>
      <th>Description</th>
      <th>Caller</th>
      <th>Assignment group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>255</th>
      <td>probleme mit laufwerk z: \laeusvjo fvaihgpx</td>
      <td>probleme mit laufwerk z: \laeusvjo fvaihgpx</td>
      <td>laeusvjo fvaihgpx</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```
df = pipeline1.preprocess_data()
```

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.



```
plt.figure(figsize=(20, 5))
df["des_lang"].value_counts().plot(kind='bar')
plt.yscale('log')
plt.ylabel("Count")
plt.xlabel("Languages")
plt.show()
```


![png](output_26_0.png)



```
print(df["des_lang"].value_counts().index[0], ": ", df["des_lang"].value_counts()[0])
```

    English :  7692



```
print("Non English Tickets: ", len(df) - df["des_lang"].value_counts()[0])
```

    Non English Tickets:  799


more english tickets are present when compared to other languages

#### Preprocessed Dataframe


```
df.head()
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
      <th>Short description</th>
      <th>Description</th>
      <th>Caller</th>
      <th>Assignment group</th>
      <th>clean_des</th>
      <th>clean_sdes</th>
      <th>des_lang</th>
      <th>sdes_lang</th>
      <th>des_has_email</th>
      <th>sdes_has_email</th>
      <th>des_has_domain</th>
      <th>sdes_has_domain</th>
      <th>des_has_url</th>
      <th>sdes_has_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>login issue</td>
      <td>-verified user details.(employee# &amp; manager na...</td>
      <td>spxjnwir pjlcoqds</td>
      <td>0</td>
      <td>(verify, user, detail, employee, manager, chec...</td>
      <td>(login, issue)</td>
      <td>English</td>
      <td>English</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>outlook</td>
      <td>\r\n\r\nreceived from: hmjdrvpb.komuaywn@gmail...</td>
      <td>hmjdrvpb komuaywn</td>
      <td>0</td>
      <td>(receive, hmjdrvpb.komuaywn@gmail.com, hello, ...</td>
      <td>(outlook)</td>
      <td>English</td>
      <td>English</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cant log in to vpn</td>
      <td>\r\n\r\nreceived from: eylqgodm.ybqkwiam@gmail...</td>
      <td>eylqgodm ybqkwiam</td>
      <td>0</td>
      <td>(receive, eylqgodm.ybqkwiam@gmail.com, hello, ...</td>
      <td>(not, log, vpn)</td>
      <td>English</td>
      <td>English</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>unable to access hr_tool page</td>
      <td>unable to access hr_tool page</td>
      <td>xbkucsvz gcpydteq</td>
      <td>0</td>
      <td>(unable, access, hr, tool, page)</td>
      <td>(unable, access, hr, tool, page)</td>
      <td>English</td>
      <td>English</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>skype error</td>
      <td>skype error</td>
      <td>owlgqjme qhcozdfx</td>
      <td>0</td>
      <td>(skype, error)</td>
      <td>(skype, error)</td>
      <td>Japanese</td>
      <td>Japanese</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


