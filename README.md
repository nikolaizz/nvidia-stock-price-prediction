# Laporan Proyek Machine Learning - Nikola Izzan Ar Rasheed

## Domain Proyek

Prediksi harga saham merupakan masalah yang sangat penting dalam dunia keuangan karena dapat membantu investor dan analis dalam mengambil keputusan investasi yang lebih tepat. Saham NVIDIA dipilih karena perusahaan ini merupakan salah satu pionir dalam teknologi grafis dan kecerdasan buatan, sehingga pergerakan harga sahamnya menarik untuk diprediksi dengan metode deep learning.

Memprediksi harga saham bukanlah hal yang mudah. Pergerakan harga saham sangat dipengaruhi oleh banyak faktor—mulai dari kondisi ekonomi global, berita perusahaan, hingga sentimen pasar yang sering kali berubah-ubah. Karena itulah, pola pergerakannya cenderung kompleks, tidak menentu, dan sulit ditebak. Model prediksi tradisional seperti regresi linier atau metode statistik lainnya memang bisa digunakan, tapi sering kali kurang akurat karena tidak mampu menangkap pola yang rumit dan hubungan jangka panjang dalam data.

Di sinilah pendekatan deep learning, khususnya Long Short-Term Memory (LSTM), menjadi sangat relevan. LSTM merupakan jenis jaringan saraf buatan yang dirancang untuk bekerja dengan data berurutan, seperti data harga saham dari waktu ke waktu. Keunggulan utama LSTM adalah kemampuannya untuk mengingat informasi penting dari masa lalu dan menggunakannya untuk memprediksi masa depan. Ini membuat LSTM cocok digunakan untuk mengenali tren dan pola jangka panjang dalam data harga saham yang sulit dilihat oleh model biasa.

Masalah ini penting untuk diselesaikan karena prediksi harga saham yang lebih akurat bisa sangat membantu para investor dan analis dalam mengambil keputusan yang lebih baik—baik itu untuk membeli, menjual, atau menyimpan saham. Di sisi lain, hal ini juga bisa digunakan untuk mengurangi risiko investasi dan memaksimalkan keuntungan. Dengan memanfaatkan teknologi seperti LSTM, kita tidak hanya mengandalkan intuisi atau tebakan, tapi bisa menggunakan data historis secara lebih cerdas dan sistematis untuk membuat prediksi yang lebih tepat.

Beberapa studi telah menunjukkan efektivitas LSTM dalam prediksi harga saham. Referensi terkait:
- Zhang, J., Lu, M., Wang, X., & Song, D. (2024). Deep learning for financial time series forecasting: A survey. arXiv. https://doi.org/10.48550/arXiv.2405.08284
- Tan, J. (2024). NVIDIA Stock Price Prediction by Machine Learning. Highlights in Business, Economics and Management, 24, 1072-1076.

## Business Understanding

### Problem Statements

- Bagaimana memprediksi harga penutupan saham NVIDIA secara akurat menggunakan data historis?
- Bagaimana mengatasi volatilitas dan pola musiman yang ada pada data harga saham?
- Bagaimana mengukur performa model prediksi agar dapat diandalkan oleh investor?

### Goals

- Mengembangkan model prediksi harga saham NVIDIA yang mampu meminimalkan kesalahan prediksi.
- Menggunakan data historis harga saham untuk mempelajari pola dan tren dengan teknik deep learning.
- Menghasilkan metrik evaluasi (MAE, RMSE, R^2) yang menunjukkan akurasi prediksi yang baik.

### Solution Statements

- Membangun model LSTM yang mampu menangkap pola temporal pada data harga saham.
- Melakukan preprocessing data, termasuk penghapusan outlier dan normalisasi fitur, untuk meningkatkan kualitas input model.
- Melakukan hyperparameter tuning seperti jumlah neuron dan dropout untuk mengurangi overfitting dan meningkatkan generalisasi model.
- Membandingkan performa model berdasarkan metrik evaluasi yang relevan untuk memilih konfigurasi terbaik.

## Data Understanding

Data yang digunakan merupakan dataset harga saham NVIDIA yang diperoleh dari Kaggle ([link dataset](https://www.kaggle.com/datasets/meharshanali/nvidia-stocks-data-2025/)). Dataset memiliki jumlah baris dan kolom awal sebanyak 6558 baris dan 7 kolom. Awalnya, dataset sudah bersih dengan tidak adanya data yang terduplikasi dan data yang memiliki missing value. Dataset berisi data harga saham dari beberapa tahun terakhir dengan kolom sebagai berikut:

- **Date**: Tanggal perdagangan (format YYYY-MM-DD)
- **Open**: Harga pembukaan saham pada hari tersebut
- **High**: Harga tertinggi saham pada hari tersebut
- **Low**: Harga terendah saham pada hari tersebut
- **Close**: Harga penutupan saham pada hari tersebut
- **Adj Close**: Harga penutupan yang disesuaikan dengan pemecahan saham dan dividen
- **Volume**: Jumlah saham yang diperdagangkan

**Exploratory Data Analysis (EDA)**

EDA dilakukan sebagai langkah awal untuk memahami pola dan karakteristik data. Beberapa hal yang dilakukan meliputi:

- Visualisasi Tren Harga Saham
![Visualisasi Tren Harga Saham](https://i.imgur.com/E9GEWjz.png)

Grafik harga Close dan Adj Close menunjukkan bahwa secara umum terdapat tren kenaikan harga saham NVIDIA seiring waktu.
Tren ini mengindikasikan adanya potensi pola jangka panjang yang bisa ditangkap oleh model time series seperti LSTM.

- Distribusi Volume Perdagangan
![Distribusi Volume Perdagangan](https://i.imgur.com/ETLGdYe.png)

Distribusi volume menunjukkan adanya ketimpangan, dengan sebagian besar hari memiliki volume perdagangan rendah hingga sedang, dan hanya sebagian kecil hari dengan volume sangat tinggi.
Hal ini penting untuk dipertimbangkan karena lonjakan volume biasanya berkaitan dengan sentimen pasar atau peristiwa penting.

- Korelasi Antar Fitur 

![Korelasi Antar Fitur](https://i.imgur.com/HdKXb5F.png)

Heatmap korelasi menunjukkan hubungan kuat antar harga saham, terutama antara Close, Adj Close, Open, High, dan Low.
Korelasi tinggi antar fitur harga mengindikasikan bahwa fitur-fitur tersebut bisa digunakan sebagai prediktor yang baik untuk memodelkan Adj Close.

- Analisis Bulanan
![Analisis Bulanan](https://i.imgur.com/fHWoMCI.png)

Rata-rata harga bulanan (Monthly Average) menunjukkan kenaikan harga saham yang signifikan dari sekitar tahun 2020 ke atas.

## Data Preparation

Proses persiapan data meliputi:

- Penghapusan outlier menggunakan metode Interquartile Range (IQR) pada fitur utama untuk mengurangi pengaruh data ekstrem.
- Penambahan fitur Simple Moving Average (SMA) 10 hari untuk menangkap tren jangka pendek.
- Normalisasi fitur menggunakan StandardScaler dan MinMaxScaler agar data berada pada rentang yang seragam, yang membantu stabilitas dan konvergensi model LSTM.
- Pembuatan sequence data dengan window 30 hari sebagai input model LSTM untuk menangkap dependensi waktu.
- Pembagian data dengan fungsi TrainTestSplit dengan pembagian 80:20.

Tahapan data preparation sangat penting dalam proyek ini karena kualitas data memiliki dampak langsung terhadap performa model deep learning yang akan dibangun. Sebelum data dimasukkan ke dalam model LSTM, dilakukan serangkaian proses pembersihan dan transformasi agar data menjadi bersih, representatif, dan relevan dengan tujuan prediksi.

Pertama, dilakukan pengecekan data kosong (null values) dan data duplikat. Ini penting untuk memastikan bahwa tidak ada entri yang hilang atau berulang yang dapat menyebabkan bias atau kesalahan dalam pelatihan model. Setelah data dipastikan lengkap dan unik, dilakukan visualisasi data dan eksplorasi awal untuk memahami karakteristik distribusi harga saham dan volume perdagangan.

Selanjutnya, dilakukan proses penghapusan outlier menggunakan metode IQR (Interquartile Range). Outlier dapat mengganggu proses pelatihan model karena nilai-nilainya yang ekstrem dapat menyebabkan model salah belajar atau terlalu fokus pada kasus-kasus yang tidak representatif. Dengan menghapus outlier dari kolom-kolom seperti Open, High, Low, Close, Adj Close, dan Volume, model dapat belajar dari data yang lebih stabil dan realistis.

Setelah data dibersihkan, dilakukan penambahan fitur teknikal seperti Simple Moving Average (SMA) 10-hari, yang dapat membantu model memahami tren jangka pendek dalam harga saham. Fitur ini juga mencerminkan kebiasaan investor yang mengandalkan pergerakan rata-rata sebagai indikator keputusan.

Kemudian, seluruh fitur numerik di-standardisasi menggunakan StandardScaler agar memiliki skala yang seragam. Ini sangat penting karena model LSTM sangat sensitif terhadap skala data. Selain itu, target output (Adj Close) juga di-normalisasi menggunakan MinMaxScaler, agar nilainya berada dalam rentang [0,1] sehingga mempercepat proses pelatihan dan meningkatkan stabilitas model.

Terakhir, dilakukan transformasi data ke dalam bentuk sekuensial. Karena LSTM membutuhkan input dalam bentuk deret waktu dengan langkah-langkah tertentu (time steps), data diubah menjadi potongan-potongan urutan (sequence) selama 30 hari ke belakang untuk memprediksi harga pada hari ke-31. Proses ini memungkinkan model belajar dari pola historis dalam jangka waktu tertentu. Yang kemudian dilanjutkan dengan proses pembagian data menggunakan fungsi train test split dengan pembagian 80% data training dan 20% data testing.

Melalui tahapan-tahapan ini, data menjadi lebih siap dan optimal untuk dilatih menggunakan model deep learning berbasis LSTM, yang pada akhirnya diharapkan mampu menghasilkan prediksi harga saham yang lebih akurat dan reliabel.

## Modeling

Model yang digunakan adalah jaringan saraf LSTM dengan arsitektur sebagai berikut:

- Satu layer LSTM dengan 64 unit neuron, menangani input sequence dengan 30 time steps dan 5 fitur.
- Dropout 20% untuk mengurangi overfitting.
- Dense layer output tunggal untuk regresi harga saham.

Model dilatih menggunakan optimizer Adam dengan fungsi loss Mean Squared Error (MSE) dan metric Mean Absolute Error (MAE). Training dilakukan selama 50 epoch dengan batch size 32 dan validasi 10% data pelatihan.

LSTM bekerja dengan menyimpan dan mengelola informasi penting dari data sekuensial melalui tiga gate utama:

- Forget Gate
Gate ini menentukan informasi apa yang harus “dilupakan” dari state sebelumnya. Misalnya, jika fluktuasi harga saham di masa lalu tidak lagi relevan, gate ini akan mengurangi kontribusinya.
- Input Gate
Gate ini menentukan informasi baru apa yang perlu ditambahkan ke memori. Dalam kasus prediksi saham, ini berarti memperhatikan perubahan harga terbaru yang bisa menjadi sinyal arah tren berikutnya.
- Output Gate
Gate ini menentukan informasi apa yang akan diteruskan ke output dan hidden state berikutnya. Output ini digunakan untuk memprediksi harga saham pada time step selanjutnya.

LSTM memiliki cell state yang berperan sebagai memori jangka panjang dan memungkinkan informasi mengalir ke depan dalam waktu yang lama tanpa mengalami peluruhan drastis, seperti yang terjadi pada RNN biasa. Hal ini sangat penting dalam prediksi saham, karena fluktuasi harga sering kali dipengaruhi oleh tren yang terjadi selama beberapa hari atau minggu sebelumnya.

Kelebihan LSTM adalah kemampuannya mengingat konteks informasi dalam jangka panjang dan menangani data deret waktu non-linear, sangat cocok untuk prediksi harga saham. Kekurangannya adalah waktu pelatihan relatif lebih lama dan membutuhkan tuning parameter yang tepat.

## Evaluation

Metrik evaluasi yang digunakan adalah:

- **Mean Absolute Error (MAE)**: rata-rata nilai absolut selisih prediksi dan nilai aktual.
- **Root Mean Squared Error (RMSE)**: akar kuadrat rata-rata kuadrat selisih prediksi dan nilai aktual, lebih sensitif terhadap outlier.
- **R-squared (R²)**: proporsi variansi data yang dapat dijelaskan oleh model.

Hasil evaluasi model LSTM:

| Metrik | Nilai  |
|--------|--------|
| MAE    | 0.011  |
| RMSE   | 0.017  |
| R²     | 0.981  |

Nilai MAE dan RMSE yang rendah menunjukkan bahwa prediksi model cukup dekat dengan nilai aktual, sementara nilai R² yang mendekati 1 menunjukkan bahwa model LSTM mampu menjelaskan sebagian besar variabilitas data harga saham. Secara keseluruhan, model dapat dikatakan memiliki performa yang sangat baik untuk melakukan prediksi harga saham NVIDIA.

Formula metrik:


![Formula Evaluasi MAE, RMSE, dan R²](https://i.imgur.com/rEGvfE0.png)

1. Mean Absolute Error (MAE)

MAE mengukur rata-rata selisih absolut antara nilai aktual (yᵢ) dan nilai prediksi (ŷᵢ):
MAE memberikan gambaran umum seberapa besar kesalahan prediksi dalam satuan asli data.
Semakin kecil nilai MAE, semakin baik performa model.

2. Root Mean Squared Error (RMSE)

RMSE mengukur akar dari rata-rata kuadrat selisih antara nilai aktual dan prediksi:
RMSE sensitif terhadap kesalahan besar (outlier).
RMSE akan selalu lebih besar atau sama dengan MAE.
Cocok digunakan ketika kesalahan besar perlu diperhatikan lebih serius.

3. R-squared (R²)

R² atau koefisien determinasi mengukur seberapa baik model menjelaskan variansi dari data aktual:
R² bernilai antara 0 hingga 1.
Semakin mendekati 1, semakin baik model dalam menjelaskan data aktual.
Nilai R² yang tinggi menunjukkan bahwa model menangkap pola tren data dengan baik.

Notasi:

yᵢ : nilai aktual ke-i
ŷᵢ : nilai prediksi ke-i
Ȳ : rata-rata dari seluruh nilai aktual
n : jumlah total data
