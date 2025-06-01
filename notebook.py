#!/usr/bin/env python
# coding: utf-8

# ## Import Library dan Data Loading

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Melakukan loading dataset

# In[2]:


df = pd.read_csv('dataset.csv')

df


# In[3]:


df.info()


# In[4]:


df.describe()


# Mengetahui jumlah data yang memiliki missing value

# In[5]:


print("Jumlah data bernilai NULL: ")
df.isnull().sum()


# Mengetahui jumlah data yang terduplikasi

# In[6]:


duplicate = df.duplicated().sum()
print(f"Jumlah data terduplikasi: {duplicate}")


# Mengetahui banyaknya bairs data dan kolom

# In[7]:


df.shape


# ## Visualisasi dan Eksplorasi Data Analisis Harga Saham NVIDIA
# Visualisasi tren harga saham dari tahun 1999-2025

# In[8]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close')
plt.plot(df['Adj Close'], label='Adj Close')
plt.legend()
plt.title('Tren Harga Penutupan NVIDIA')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.show()


# Distribusi volume perdagangan (jumlah perdagangan saham)

# In[9]:


plt.figure(figsize=(12,5))
sns.histplot(df['Volume'], bins=50, kde=True)
plt.title('Distribusi Volume Perdagangan')
plt.show()


# Heatmap korelasi antar fitur

# In[10]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korelasi Antar Fitur')
plt.show()


# Melihat tren rata-rata harga per bulan

# In[11]:


monthly = df['Close'].resample('M').mean()
plt.figure(figsize=(14,5))
monthly.plot()
plt.title('Rata-rata Harga Bulanan')
plt.show()


# Melihat outliers pada data

# In[12]:


plt.figure(figsize=(12,5))
sns.boxplot(data=df[['Open','High','Low','Close','Adj Close']])
plt.title('Boxplot Harga Harian')
plt.show()


# ## Data Preprocessing

# Melakukan penghapusan outliers

# In[13]:


def remove_outliers_iqr(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        after = df_clean.shape[0]
        print(f"{col}: Dihapus {before - after} outlier")
    return df_clean

cols_outlier = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

df = remove_outliers_iqr(df, cols_outlier)

print("Setelah penghapusan outliers:", df.shape)


# Melakukan standarisasi pada data

# In[14]:


df['SMA_10'] = df['Close'].rolling(window=10).mean()

features = df[['Open','High','Low','Close','Volume']].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

y = df['Adj Close'].values.reshape(-1, 1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)


# ## Split Data dan Pemodelan

# Modelling menggunakan LSTM yang diawali dengan pembuatan data sekuensial

# In[15]:


def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps=30)

split_index = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)


# ## Evaluasi Model

# In[16]:


y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("Evaluasi LSTM:")
print(f"MAE   : {mae:.3f}")
print(f"RMSE  : {rmse:.3f}")
print(f"R^2   : {r2:.3f}")


# ## Prediksi dan Visualisasi

# In[17]:


hasil_prediksi = pd.DataFrame({
    'Tanggal': df.index[-len(y_test):],
    'Harga Aktual': y_true.flatten(),
    'Harga Prediksi': y_pred.flatten()
})
hasil_prediksi.set_index('Tanggal', inplace=True)

plt.figure(figsize=(14,6))
plt.plot(hasil_prediksi['Harga Aktual'], label='Harga Aktual', linewidth=2)
plt.plot(hasil_prediksi['Harga Prediksi'], label='Harga Prediksi', linestyle='--')
plt.title('Prediksi Harga Saham vs Harga Aktual')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

