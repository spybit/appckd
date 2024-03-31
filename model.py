import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Importing data set
df = pd.read_csv("kidney_disease.csv")

# Filling Null Values
df['age'] = df['age'].fillna(df['age'].mean())
df['bp'] = df['bp'].fillna(df['bp'].mean())
df['sg'] = df['sg'].fillna(df['sg'].mean())
df['al'] = df['al'].fillna(df['al'].mean())
df['su'] = df['su'].fillna(df['su'].mean())
df['bgr'] = df['bgr'].fillna(df['bgr'].mean())
df['bu'] = df['bu'].fillna(df['bu'].mean())
df['sc'] = df['sc'].fillna(df['sc'].mean())
df['sod'] = df['sod'].fillna(df['sod'].mean())
df['pot'] = df['pot'].fillna(df['pot'].mean())
df['hemo'] = df['hemo'].fillna(df['hemo'].mean())
df['rbc'] = df['rbc'].fillna(df['rbc'].mode()[0])
df['pc'] = df['pc'].fillna(df['pc'].mode()[0])
df['pcc'] = df['pcc'].fillna(df['pcc'].mode()[0])
df['ba'] = df['ba'].fillna(df['ba'].mode()[0])
df['pcv'] = df['pcv'].fillna(df['pcv'].mode()[0])
df['wc'] = df['wc'].fillna(df['wc'].mode()[0])
df['rc'] = df['rc'].fillna(df['rc'].mode()[0])
df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
df['dm'] = df['dm'].fillna(df['dm'].mode()[0])
df['cad'] = df['cad'].fillna(df['cad'].mode()[0])
df['appet'] = df['appet'].fillna(df['appet'].mode()[0])
df['pe'] = df['pe'].fillna(df['pe'].mode()[0])
df['ane'] = df['ane'].fillna(df['ane'].mode()[0])
df['classification'] = df['classification'].fillna(df['classification'].mode()[0])
df['classification'] = df['classification'].str.replace("\t", "")

#handling categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.rbc = le.fit_transform(df.rbc)
df.pc = le.fit_transform(df.pc)
df.pcc = le.fit_transform(df.pcc)
df.ba = le.fit_transform(df.ba)
df.htn = le.fit_transform(df.htn)
df.dm = le.fit_transform(df.dm)
df.cad = le.fit_transform(df.cad)
df.appet = le.fit_transform(df.appet)
df.pe = le.fit_transform(df.pe)
df.ane = le.fit_transform(df.ane)
df.rc = le.fit_transform(df.rc)
df.wc = le.fit_transform(df.wc)
df.classification = le.fit_transform(df.classification)
df = df[['bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'rc', 'htn', 'dm', 'appet', 'pe', 'classification']]
# spliting
x = df.drop(columns=['classification'], axis=1)
y = df['classification']

# solving imblancing
ckd = df[df['classification']==0]
notckd = df[df['classification']==1]
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
smk = SMOTETomek(random_state=42)
x_res, y_res = smk.fit_resample(x, y)

# Traning Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.25, random_state=42)

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
model7 = AdaBoostClassifier()
model7.fit(x_train, y_train)

#print accuracy
#e7 = model7.score(x_test, y_test)*100
#print("Accuracy is", f"{e7:.5f}")

# make pickle file
pickle.dump(model7, open("model.pkl", "wb"))