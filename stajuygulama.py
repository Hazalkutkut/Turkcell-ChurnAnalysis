import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#veri setini yüklemek
df_turn1 = pd.read_excel('turnover_2.xlsx')
print(df_turn1.head())

#verinin boyutunu görmek
df_turn1.shape

#her sütunun veri tipini görmek
df_turn1.info()

#istatiksel veriler
df_turn1.describe()

df_turn1.describe(include = 'object')

#boş veri var mı?
eksik_degerler = df_turn1.isna().sum()
print(eksik_degerler[eksik_degerler > 0])

# Tekrar eden satırları kontrol et ve kaldır
df_turn1 = df_turn1.drop_duplicates()

import matplotlib.pyplot as plt
import seaborn as sns

# Turnover dağılımı
plt.figure(figsize=(6,4))
sns.countplot(x='Turnover', data=df_turn1, palette="Set2")
plt.title("Turnover Dağılımı")
plt.show()

# Yaş ve Turnover arasındaki ilişki
plt.figure(figsize=(8,6))
sns.boxplot(x='Turnover', y='Age', data=df_turn1, palette="Set2")
plt.title("Turnover ve Yaş İlişkisi")
plt.show()

# Kategorik değişkenleri sayısal forma çevirelim
from sklearn.preprocessing import LabelEncoder

# Kategorik sütunları etiketleyelim
label_encoder = LabelEncoder()
categorical_columns = ['Turnover']

df_turn1[categorical_columns] = df_turn1[categorical_columns].apply(label_encoder.fit_transform)

# Güncellenmiş veriyi görelim
df_turn1.head()

# Create subplots for kde plots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Select only numerical features for kde plots
numerical_features = df_turn1.select_dtypes(include=['number']).columns

for ax, col in zip(axes.flatten(), numerical_features):
    sns.kdeplot(data=df_turn1, x=col, fill=True, linewidth=2, hue='Turnover', ax=ax, palette = {0: '#009c05', 1: 'darkorange'})
    ax.set_title(f'{col} vs Target')

axes[2,1].axis('off') # Turn off the last subplot if not used
plt.suptitle('Distribution of Continuous Features by Target', fontsize=22)
plt.tight_layout()
plt.show()

# List of categorical features
categorical_columns = ['Gender', 'MaritalStatus', 'Turnover', 'Travelling', 'Vertical', 'EducationField', 'OverTime', 'Role']

# Initialize the plot
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Plot each feature
for i, ax in enumerate(axes.flatten()):
    sns.countplot(x=categorical_columns[i], hue='Turnover', data=df_turn1, ax=ax, palette={0: '#009c05', 1: 'darkorange'})
    ax.set_title(categorical_columns[i])
    ax.set_ylabel('Count')
    ax.set_xlabel('')
    ax.legend(title='Left', loc='upper right')

plt.suptitle('Distribution of Categorical Features by Target', fontsize=22)
plt.tight_layout()
plt.show()

# Kategorik değişkenleri sayısal forma çevirelim
from sklearn.preprocessing import LabelEncoder

# Kategorik sütunları etiketleyelim
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'MaritalStatus', 'Travelling', 'Vertical', 'EducationField', 'OverTime', 'Role']

df_turn1[categorical_columns] = df_turn1[categorical_columns].apply(label_encoder.fit_transform)

# Güncellenmiş veriyi görelim
df_turn1.head()

df_turn1.corr()

# Örnek: Korelasyon ısı haritası
plt.figure(figsize=(20,16))
sns.heatmap(df_turn1.corr(), annot=True, cmap='coolwarm')
plt.show()

# Örnek: Hedef değişkenin dağılımı
sns.countplot(x='Turnover', data=df_turn1)
plt.show()

# test ve train olarak ayırma
X = df_turn1.drop(columns=['Turnover'])
y = df_turn1['Turnover']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

CV = [] #çarpaz doğrulama
R2_train = []
R2_test = []

def turn1_model(model,model_name):
    # modeli eğitme
    model.fit(X_train,y_train)

    # Train setinin R2 skoru
    y_pred_train = model.predict(X_train)
    R2_train_model = r2_score(y_train,y_pred_train)
    R2_train.append(round(R2_train_model,2))

    # Test setinin R2 skoru
    y_pred_test = model.predict(X_test)
    R2_test_model = r2_score(y_test,y_pred_test)
    R2_test.append(round(R2_test_model,2))

    # çarpraz doğrulama
    cross_val = cross_val_score(model ,X_train ,y_train ,cv=5)
    cv_mean = cross_val.mean()
    CV.append(round(cv_mean,2))

    # sonuçları yazdırma
    print("Train R2-score :",round(R2_train_model,2))
    print("Test R2-score :",round(R2_test_model,2))
    print("Train CV scores :",cross_val)
    print("Train CV mean :",round(cv_mean,2))

    # grafik
    # Residual Plot of train data
    fig, ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].set_title('Residual Plot of Train samples')
    sns.distplot((y_train-y_pred_train),hist = False,ax = ax[0])
    ax[0].set_xlabel('y_train - y_pred_train')

    # Y_test ve Y_train scatter plot
    ax[1].set_title('y_test vs y_pred_test')
    ax[1].scatter(x = y_test, y = y_pred_test)
    ax[1].set_xlabel('y_test')
    ax[1].set_ylabel('y_pred_test')

    plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
turn1_model(lr,"Linear_regressor.pkl")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lg = LogisticRegression()
turn1_model(lg,"Logistic_regressor.pkl")

from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

# Creating Ridge model object
rg = Ridge()
# range of alpha
alpha = np.logspace(-3,3,num=14)

# Creating RandomizedSearchCV to find the best estimator of hyperparameter
rg_rs = RandomizedSearchCV(estimator = rg, param_distributions = dict(alpha=alpha))

turn1_model(rg_rs,"ridge.pkl")

from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV

ls = Lasso()
alpha = np.logspace(-3,3,num=14) # range for alpha

ls_rs = RandomizedSearchCV(estimator = ls, param_distributions = dict(alpha=alpha))
turn1_model(ls_rs,"lasso.pkl")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor()

# Number of trees in Random forest
n_estimators=list(range(500,1000,100))
# Maximum number of levels in a tree
max_depth=list(range(4,9,4))
# Minimum number of samples required to split an internal node
min_samples_split=list(range(4,9,2))
# Minimum number of samples required to be at a leaf node.
min_samples_leaf=[1,2,5,7]
# Number of fearures to be considered at each split
max_features=['auto','sqrt']

# Hyperparameters dict
param_grid = {"n_estimators":n_estimators,
              "max_depth":max_depth,
              "min_samples_split":min_samples_split,
              "min_samples_leaf":min_samples_leaf,
              "max_features":max_features}

rf_rs = RandomizedSearchCV(estimator = rf, param_distributions = param_grid)
turn1_model(rf_rs,'random_forest.pkl')

# Özellik önemlerini alalım
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Access the fitted RandomForestRegressor model using best_estimator_
rf_model = rf_rs.best_estimator_  # Assuming rf_rs is defined in a previous cell

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Özellik önemlerini görselleştirelim
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
plt.title("Random Forest Özellik Önemleri")
plt.xlabel("Özellik Önemi Skoru")
plt.ylabel("Özellikler")
plt.show()

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

gb = GradientBoostingRegressor()

# Rate at which correcting is being made
learning_rate = [0.001, 0.01, 0.1, 0.2]
# Number of trees in Gradient boosting
n_estimators=list(range(500,1000,100))
# Maximum number of levels in a tree
max_depth=list(range(4,9,4))
# Minimum number of samples required to split an internal node
min_samples_split=list(range(4,9,2))
# Minimum number of samples required to be at a leaf node.
min_samples_leaf=[1,2,5,7]
# Number of fearures to be considered at each split
max_features=['auto','sqrt']

# Hyperparameters dict
param_grid = {"learning_rate":learning_rate,
              "n_estimators":n_estimators,
              "max_depth":max_depth,
              "min_samples_split":min_samples_split,
              "min_samples_leaf":min_samples_leaf,
              "max_features":max_features}

gb_rs = RandomizedSearchCV(estimator = gb, param_distributions = param_grid)

turn1_model(gb_rs,"gradient_boosting.pkl")

print(rf_rs.best_estimator_)

Technique = ["LinearRegression","LogisticRegression","Ridge","Lasso","RandomForestRegressor","Gradient Boosting"]
results=pd.DataFrame({'Model': Technique,'R Squared(Train)': R2_train,'R Squared(Test)': R2_test,'CV score mean(Train)': CV})
display(results)

from sklearn.metrics import roc_auc_score, roc_curve

# Assuming y_pred contains predictions for y_test
y_pred = gb_rs.predict(X_test) # Make sure to have defined X_test elsewhere in your code, or replace with appropriate variable

# AUC skorunu hesaplayalım
auc_score = roc_auc_score(y_test, y_pred)  # Changed y_train to y_pred
print(f"AUC Skoru: {auc_score}")

# ROC eğrisini çizelim
fpr, tpr, thresholds = roc_curve(y_test, y_pred)  # Changed y_train to y_pred
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Hiperparametre grid'ini oluşturma
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Random Forest modelini başlatalım
rf = RandomForestClassifier(random_state=42)

# GridSearchCV ile parametre optimizasyonu
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# En iyi parametreleri gösterelim
print("En iyi parametreler:", grid_search.best_params_)

# En iyi model ile test setinde performans ölçümü
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"En iyi model doğruluk skoru: {accuracy}")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Hiperparametre aralığı
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'bootstrap': [True, False]
}

# Random Forest modelini başlatalım
rf = RandomForestClassifier(random_state=42)

# RandomizedSearchCV ile parametre optimizasyonu
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc', random_state=42)
random_search.fit(X_train, y_train)

# En iyi parametreleri gösterelim
print("En iyi parametreler:", random_search.best_params_)

# En iyi model ile test setinde performans ölçümü
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"En iyi model doğruluk skoru: {accuracy}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Karmaşıklık matrisini oluştur
cm = confusion_matrix(y_test, y_pred)

# Görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('Karmaşıklık Matrisi')
plt.show()

import joblib

# Modeli kaydet
joblib.dump(best_rf, 'optimized_random_forest_model.pkl')