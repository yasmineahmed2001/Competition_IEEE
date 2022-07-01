{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 518,
   "id": "46bc929f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import linear_model\n",
    "from sklearn import metrics\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import MultiLabelBinarizer\n",
    "from datetime import datetime\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error as MSE\n",
    "import time\n",
    "from sklearn import datasets\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import GradientBoostingRegressor\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 519,
   "id": "4be0170d",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(r'D:\\\\competiton machine\\\\train.csv')\n",
    "data2 = pd.read_csv(r'D:\\\\competiton machine\\\\test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 520,
   "id": "bc6b2dbf",
   "metadata": {},
   "outputs": [],
   "source": [
    "label = LabelEncoder()\n",
    "m = MultiLabelBinarizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 521,
   "id": "07c32c6d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Lenovo\\AppData\\Local\\Temp/ipykernel_17240/2883598858.py:30: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.\n",
      "  data['cast']=data['cast'].str.replace(\"|\",\" \")\n",
      "C:\\Users\\Lenovo\\AppData\\Local\\Temp/ipykernel_17240/2883598858.py:61: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.\n",
      "  data2['cast']=data2['cast'].str.replace(\"|\",\" \")\n"
     ]
    }
   ],
   "source": [
    "#colum Unnamed: 0   \n",
    "data = data.iloc[: , 1:]\n",
    "data2 = data2.iloc[: , 1:]\n",
    "\n",
    "#colum imbd_id\n",
    "data['imdb_id']= data['imdb_id'].str.replace(\"tt\",\"\").astype(object).astype(float)\n",
    "data['imdb_id'] = (data['imdb_id']-data['imdb_id'].min())/(data['imdb_id'].max()-data['imdb_id'].min())\n",
    "data['imdb_id'].fillna(data['imdb_id'].mean() , inplace = True)\n",
    "#colum popularity\n",
    "data['popularity']=data['popularity'].round(3)\n",
    "\n",
    "#budget\n",
    "data['budget']=label.fit_transform(data['budget'])\n",
    "data['budget'] = (data['budget']-data['budget'].min())/(data['budget'].max()-data['budget'].min())\n",
    "\n",
    "#revenue\n",
    "data['revenue']=label.fit_transform(data['revenue'])\n",
    "data['revenue'] = (data['revenue']-data['revenue'].min())/(data['revenue'].max()-data['revenue'].min())\n",
    "\n",
    "\n",
    "#director\n",
    "data['director']=label.fit_transform(data['director'])\n",
    "data['director'] = (data['director']-data['director'].min())/(data['director'].max()-data['director'].min())\n",
    "\n",
    "#homepage\n",
    "data['homepage']=label.fit_transform(data['homepage'])\n",
    "data['homepage'] = (data['homepage']-data['homepage'].min())/(data['homepage'].max()-data['homepage'].min())\n",
    "\n",
    "#cast\n",
    "data['cast']=data['cast'].str.replace(\"|\",\" \")\n",
    "data['cast']=label.fit_transform(data['cast'])\n",
    "data['cast'] = (data['cast']-data['cast'].min())/(data['cast'].max()-data['cast'].min())\n",
    "\n",
    "\n",
    "\n",
    "#data2\n",
    "#colum imbd_id\n",
    "data2['imdb_id']= data2['imdb_id'].str.replace(\"tt\",\"\").astype(object).astype(float)\n",
    "data2['imdb_id'] = (data2['imdb_id']-data2['imdb_id'].min())/(data2['imdb_id'].max()-data2['imdb_id'].min())\n",
    "data2['imdb_id'].fillna(data2['imdb_id'].mean() , inplace = True)\n",
    "#colum popularity\n",
    "data2['popularity']=data2['popularity'].round(3)\n",
    "\n",
    "#budget\n",
    "data2['budget']=label.fit_transform(data2['budget'])\n",
    "data2['budget'] = (data2['budget']-data2['budget'].min())/(data2['budget'].max()-data2['budget'].min())\n",
    "\n",
    "#revenue\n",
    "data2['revenue']=label.fit_transform(data2['revenue'])\n",
    "data2['revenue'] = (data2['revenue']-data2['revenue'].min())/(data2['revenue'].max()-data2['revenue'].min())\n",
    "\n",
    "#homepage\n",
    "data2['homepage']=label.fit_transform(data2['homepage'])\n",
    "data2['homepage'] = (data2['homepage']-data2['homepage'].min())/(data2['homepage'].max()-data2['homepage'].min())\n",
    "\n",
    "#director\n",
    "data2['director']=label.fit_transform(data2['director'])\n",
    "data2['director'] = (data2['director']-data2['director'].min())/(data2['director'].max()-data2['director'].min())\n",
    "\n",
    "#cast\n",
    "data2['cast']=data2['cast'].str.replace(\"|\",\" \")\n",
    "data2['cast']=label.fit_transform(data2['cast'])\n",
    "data2['cast'] = (data2['cast']-data2['cast'].min())/(data2['cast'].max()-data2['cast'].min())\n",
    "\n",
    "\n",
    "\n",
    "del data['release_date']\n",
    "del data['overview']\n",
    "del data['original_title']\n",
    "del data['keywords']\n",
    "del data['tagline']\n",
    "del data ['budget_adj']\n",
    "del data ['revenue_adj']\n",
    "del data ['genres']\n",
    "del data ['production_companies']\n",
    "\n",
    "\n",
    "del data2['release_date']\n",
    "del data2['overview']\n",
    "del data2['original_title']\n",
    "del data2['keywords']\n",
    "del data2['tagline']\n",
    "del data2 ['budget_adj']\n",
    "del data2 ['revenue_adj']\n",
    "del data2 ['genres']\n",
    "del data2 ['production_companies']\n",
    "del data['vote_count']\n",
    "del data['release_year']\n",
    "del data2['vote_count']\n",
    "del data2['release_year']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 522,
   "id": "b2165203",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.dropna(how='any',inplace=True)\n",
    "data2.dropna(how='any',inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 523,
   "id": "752a4bac",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train=data.iloc[:,0:8]\n",
    "Y_train=data['vote_average']\n",
    "\n",
    "X_test = data2.iloc[:, 0:8]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 524,
   "id": "ef479a10",
   "metadata": {},
   "outputs": [],
   "source": [
    "sc = StandardScaler()\n",
    "\n",
    "X_train_std = sc.fit_transform(X_train)\n",
    "X_test_std = sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 525,
   "id": "4f54d502",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GradientBoostingRegressor(learning_rate=0.01, max_depth=1, min_samples_split=3,\n",
       "                          n_estimators=50)"
      ]
     },
     "execution_count": 525,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gbr_params = {'n_estimators':50,'max_depth': 1, 'min_samples_split':3, 'learning_rate': 0.01, 'loss': 'ls'}\n",
    "gbr = GradientBoostingRegressor(**gbr_params)\n",
    "gbr.fit(X_train_std, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 526,
   "id": "67ea6189",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test = gbr.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 527,
   "id": "426720b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Lenovo\\AppData\\Local\\Temp/ipykernel_17240/4248540577.py:5: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  da['Predicted'][i]=y_test[i]\n"
     ]
    }
   ],
   "source": [
    "da= pd.read_csv(r'D:\\\\competiton machine\\\\Sample_submission.csv')\n",
    "#map them in a dictionary\n",
    "da['Predicted']=da['Predicted'].astype(float)\n",
    "for i in range(0,1500):\n",
    "    da['Predicted'][i]=y_test[i]\n",
    "da.to_csv('Sample_submission.csv')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "06a79995",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2a46702",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "701da3d4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6962098",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e0db649",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
