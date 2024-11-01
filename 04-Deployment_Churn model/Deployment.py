{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "80de20ff-1d26-40a8-b72e-b9fe2f734764",
   "metadata": {},
   "source": [
    "## Code Deployment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "27136b18-e3da-4e65-a1ed-8213691060f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import KFold\n",
    "\n",
    "from sklearn.feature_extraction import DictVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import roc_auc_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0cc3b738-a77c-4516-aa79-c95fa2f847cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ac1e3496-446d-48e8-b4da-2a4bfd206af7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(data)\n",
    "\n",
    "df.columns = df.columns.str.lower().str.replace(' ', '_')\n",
    "\n",
    "categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)\n",
    "\n",
    "for c in categorical_columns:\n",
    "    df[c] = df[c].str.lower().str.replace(' ', '_')\n",
    "\n",
    "df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')\n",
    "df.totalcharges = df.totalcharges.fillna(0)\n",
    "\n",
    "df.churn = (df.churn == 'yes').astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "bc9f26ce-c306-446b-bdc9-3ea7b0d517c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "679caa95-762a-420c-9568-8588aa8db16f",
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical = ['tenure', 'monthlycharges', 'totalcharges']\n",
    "\n",
    "categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',\n",
    "        'phoneservice', 'multiplelines', 'internetservice',\n",
    "       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',\n",
    "       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',\n",
    "       'paymentmethod']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "45c80799-304f-4dde-af19-497a21baf25b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(df_train, y_train, C=1.0):\n",
    "    dicts = df_train[categorical + numerical].to_dict(orient='records')\n",
    "    \n",
    "    dv = DictVectorizer(sparse=False)\n",
    "    X_train = dv.fit_transform(dicts)\n",
    "    \n",
    "    model = LogisticRegression(C=C, solver='liblinear', random_state=1, max_iter=1000)\n",
    "    model.fit(X_train, y_train)\n",
    "\n",
    "    return dv, model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "61140f3e-ffc0-4c88-99eb-551a8a755002",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(df, dv, model):\n",
    "    dicts = df[categorical + numerical].to_dict(orient='records')\n",
    "\n",
    "    X = dv.transform(dicts)\n",
    "    y_pred = model.predict_proba(X)[:, 1]\n",
    "\n",
    "    return y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f225b681-fe1a-4b47-8d9e-0d99a73c1993",
   "metadata": {},
   "outputs": [],
   "source": [
    "C = 1.0\n",
    "n_splits = 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "67c50ddc-8b29-4c19-a58b-3f190495652c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C=1.0 0.841 +- 0.007\n"
     ]
    }
   ],
   "source": [
    "kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)\n",
    "    \n",
    "scores = []\n",
    "\n",
    "for train_idx, val_idx in kfold.split(df_full_train):\n",
    "    df_train = df_full_train.iloc[train_idx]\n",
    "    df_val = df_full_train.iloc[val_idx]\n",
    "    \n",
    "    y_train = df_train.churn.values\n",
    "    y_val = df_val.churn.values    \n",
    "        \n",
    "    dv, model = train(df_train, y_train, C=C)\n",
    "    y_pred = predict(df_val, dv, model)\n",
    "    \n",
    "    auc = roc_auc_score(y_val, y_pred)\n",
    "    scores.append(auc)\n",
    "\n",
    "print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "bd8a5959-8a0d-4233-9014-f644ae58e2f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[np.float64(0.8423240260300963),\n",
       " np.float64(0.8453247086478611),\n",
       " np.float64(0.8337066024483243),\n",
       " np.float64(0.8323627454115241),\n",
       " np.float64(0.8521736060995889)]"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "de0a342d-0e84-458a-ac5c-45c5e45fa914",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(0.8579400803839363)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)\n",
    "y_pred = predict(df_test, dv, model)\n",
    "\n",
    "y_test = df_test.churn.values\n",
    "auc = roc_auc_score(y_test, y_pred)\n",
    "auc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e8bfe111-7b41-48f4-a746-7be95ee06c7e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "191a5b88-b54c-41b1-81ac-1762df3c98d1",
   "metadata": {},
   "source": [
    "### Save the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7b269d69-41e0-472c-a94c-e902ff62746b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8e477c85-3f3a-4841-b9ac-0021a33813de",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'model_C=1.0.bin'"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output_file = f'model_C={C}.bin'\n",
    "output_file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "338e8b30-e5e4-41f4-b8f0-5d580004aabc",
   "metadata": {},
   "outputs": [],
   "source": [
    "f_out = open('output_file', 'wb')\n",
    "pickle.dump((dv, model), f_out)\n",
    "f_out.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c4cedd88-8719-426b-a5db-2fef8db4cb9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(output_file, 'wb') as f_out:\n",
    "    pickle.dump((dv, model), f_out)\n",
    "    # do stuff\n",
    "\n",
    "# do other stuff - the file is automatically closed here."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2afac21c-2ff9-4d1c-82e8-3cf42556efaf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "7d24453b-697d-4baf-b802-639ddecbf095",
   "metadata": {},
   "source": [
    "### Load the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "703fc1f7-5f1b-48ab-bc23-03aa64c9ab85",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a6c24564-11f7-434b-b8f1-e8bf93a4ca01",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_file = 'model_C=1.0.bin'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "21c51278-7883-4e5c-aad0-c1dda102f948",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(output_file, 'rb') as f_in:\n",
    "    (dv, model) = pickle.load(f_in)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "dc17bded-8f0d-4720-8638-af8ab781f7cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(DictVectorizer(sparse=False),\n",
       " LogisticRegression(max_iter=1000, random_state=1, solver='liblinear'))"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dv, model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "625f13f1-438d-4598-90a5-b3bccecd2918",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "ce5d0d5c-3e91-41e0-b14f-3c26d02ea1be",
   "metadata": {},
   "outputs": [],
   "source": [
    "customer = {\n",
    "    'gender': 'female',\n",
    "    'seniorcitizen': 0,\n",
    "    'partner': 'yes',\n",
    "    'dependents': 'no',\n",
    "    'phoneservice': 'no',\n",
    "    'multiplelines': 'no_phone_service',\n",
    "    'internetservice': 'dsl',\n",
    "    'onlinesecurity': 'no',\n",
    "    'onlinebackup': 'yes',\n",
    "    'deviceprotection': 'no',\n",
    "    'techsupport': 'no',\n",
    "    'streamingtv': 'no',\n",
    "    'streamingmovies': 'no',\n",
    "    'contract': 'month-to-month',\n",
    "    'paperlessbilling': 'yes',\n",
    "    'paymentmethod': 'electronic_check',\n",
    "    'tenure': 1,\n",
    "    'monthlycharges': 29.85,\n",
    "    'totalcharges': 29.85\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "8329595a-ca94-41ff-b091-93cfd19cecd3",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dv.transform([customer])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "1944d6d5-fdeb-4510-bb50-ccceb203a396",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(0.6433010959417496)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict_proba(X)[0, 1]"
   ]
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
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
