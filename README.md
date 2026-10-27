# Customer Churn Prediction

## Deskripsi Proyek

**Customer Churn Prediction** merupakan proyek Machine Learning yang bertujuan untuk memprediksi kemungkinan pelanggan berhenti berlangganan (*customer churn*) berdasarkan data historis pelanggan. Sistem menganalisis karakteristik pelanggan, informasi layanan, durasi berlangganan, serta data tagihan untuk mengidentifikasi pola yang berkaitan dengan perpindahan pelanggan.

Dengan menerapkan pendekatan *Supervised Learning*, proyek ini berfokus pada pengembangan dan evaluasi model klasifikasi yang mampu membedakan pelanggan yang berhenti berlangganan dengan pelanggan yang tetap menggunakan layanan.

Hasil prediksi diharapkan dapat mendukung pengambilan keputusan berbasis data, khususnya dalam mengidentifikasi pelanggan berisiko dan menyusun strategi retensi yang lebih terarah.

## Latar Belakang dan Permasalahan

Customer churn merupakan salah satu tantangan utama bagi bisnis berbasis langganan. Kehilangan pelanggan dapat berdampak pada penurunan pendapatan berulang, peningkatan kebutuhan akuisisi pelanggan baru, serta berkurangnya potensi pendapatan jangka panjang.

Proyek ini berfokus pada beberapa permasalahan bisnis berikut:

- **Penurunan Pendapatan Berulang:** Berhentinya pelanggan berlangganan dapat mengurangi pendapatan rutin perusahaan.
- **Biaya Akuisisi Pelanggan:** Upaya memperoleh pelanggan baru membutuhkan sumber daya pemasaran dan operasional. Mempertahankan pelanggan yang sudah ada dapat membantu mengoptimalkan sumber daya tersebut.
- **Keterbatasan Deteksi Dini:** Tanpa analisis prediktif, perusahaan dapat mengalami kesulitan mengenali pelanggan yang berpotensi berhenti sebelum keputusan tersebut terjadi.
- **Kebutuhan Analisis Berbasis Data:** Informasi historis pelanggan dapat dimanfaatkan untuk menemukan pola churn dan mendukung penyusunan strategi retensi.

Tujuan utama proyek ini adalah membangun serta membandingkan beberapa model klasifikasi untuk mengidentifikasi pelanggan yang berpotensi mengalami churn.

## Dataset

Proyek ini menggunakan dataset **Telco Customer Churn** yang tersedia melalui Kaggle.

### Ringkasan Dataset

- **Nama Dataset:** Telco Customer Churn
- **Jumlah Data:** 7.043 baris
- **Jumlah Kolom:** 21 kolom, termasuk variabel target
- **Variabel Target:** `Churn`
- **Jenis Permasalahan:** Klasifikasi biner
- **Pendekatan Pembelajaran:** Supervised Learning

### Kategori Fitur

#### 1. Demografi Pelanggan

- `gender`: Jenis kelamin pelanggan.
- `SeniorCitizen`: Indikator apakah pelanggan termasuk kelompok warga senior.
- `Partner`: Indikator apakah pelanggan memiliki pasangan.
- `Dependents`: Indikator apakah pelanggan memiliki tanggungan.

#### 2. Layanan yang Digunakan

- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

Fitur-fitur tersebut menggambarkan jenis layanan telekomunikasi dan layanan tambahan yang digunakan pelanggan.

#### 3. Informasi Akun dan Pembayaran

- `tenure`: Durasi pelanggan berlangganan dalam satuan bulan.
- `Contract`: Jenis kontrak berlangganan, meliputi bulanan, satu tahun, dan dua tahun.
- `PaperlessBilling`: Indikator penggunaan tagihan elektronik tanpa kertas.
- `PaymentMethod`: Metode pembayaran yang digunakan pelanggan.
- `MonthlyCharges`: Biaya layanan bulanan.
- `TotalCharges`: Akumulasi biaya layanan selama periode berlangganan.

#### 4. Variabel Target

- `Churn`: Status pelanggan, dengan nilai `Yes` untuk pelanggan yang berhenti berlangganan dan `No` untuk pelanggan yang tetap menggunakan layanan.

Variabel `Churn` digunakan sebagai label dalam proses pelatihan dan evaluasi model klasifikasi.

## Metodologi

Pengembangan model dilakukan melalui tahapan pengolahan data, transformasi fitur, pelatihan model, evaluasi performa, dan interpretasi hasil prediksi.

### 1. Pra-pemrosesan Data (*Data Preprocessing*)

Tahap pra-pemrosesan bertujuan memastikan data memiliki format yang sesuai sebelum digunakan untuk melatih model.

- **Pembersihan Data (*Data Cleaning*):** Memeriksa nilai kosong, ketidakkonsistenan data, dan format variabel yang tidak sesuai.
- **Konversi Tipe Data:** Mengubah kolom `TotalCharges` menjadi format numerik serta menangani nilai yang tidak dapat dikonversi.
- **Encoding Fitur Kategorikal:** Menggunakan *One-Hot Encoding* untuk mengubah variabel kategorikal menjadi representasi numerik.
- **Standardisasi Fitur (*Feature Scaling*):** Menggunakan `StandardScaler` untuk menyamakan skala fitur numerik, terutama pada algoritma yang sensitif terhadap perbedaan skala.
- **Penanganan Ketidakseimbangan Kelas (*Class Imbalance*):** Mengevaluasi distribusi kelas target dan mempertimbangkan teknik seperti SMOTE apabila diperlukan.

Seluruh proses transformasi yang mempelajari parameter dari data harus disesuaikan hanya menggunakan data pelatihan (*training set*). Apabila SMOTE digunakan, proses oversampling juga harus dibatasi pada data pelatihan untuk mencegah *data leakage*.

### 2. Pemodelan Machine Learning

Lima algoritma klasifikasi digunakan sebagai kandidat model untuk dibandingkan berdasarkan performa pada data evaluasi.

#### 1.Logistic Regression

Algoritma klasifikasi yang memodelkan probabilitas suatu observasi termasuk ke dalam kelas tertentu menggunakan fungsi logistik.

**Tujuan penggunaan:** Menjadi model baseline untuk memperoleh tolok ukur performa awal serta membantu memahami hubungan antara fitur masukan dan probabilitas churn.

#### 2.K-Nearest Neighbors (KNN)

Algoritma klasifikasi berbasis jarak yang menentukan kelas suatu observasi berdasarkan label dari sejumlah tetangga terdekat dalam ruang fitur.

**Tujuan penggunaan:** Mengevaluasi apakah pelanggan dengan karakteristik layanan, durasi berlangganan, dan pola tagihan yang serupa memiliki kecenderungan churn yang serupa.

#### 3.Support Vector Classifier (SVC)

Algoritma klasifikasi yang mencari batas keputusan optimal dengan memaksimalkan margin pemisah antarkelas. Fungsi kernel juga dapat digunakan untuk menangani pola pemisahan nonlinier.

**Tujuan penggunaan:** Mengevaluasi efektivitas klasifikasi berbasis margin dalam membedakan pelanggan churn dan non-churn.

#### 4.Decision Tree Classifier

Algoritma berbasis pohon keputusan yang membagi data secara bertahap berdasarkan aturan pada fitur tertentu. Pemilihan pemisahan dapat menggunakan kriteria seperti *Gini Impurity* atau *Entropy*.

**Tujuan penggunaan:** Menghasilkan model dengan alur keputusan yang relatif mudah diinterpretasikan untuk memahami karakteristik pelanggan yang berkaitan dengan churn.

#### 5.Random Forest Classifier

Algoritma *Ensemble Learning* yang menggabungkan prediksi dari sejumlah pohon keputusan yang dilatih menggunakan variasi sampel dan subset fitur.

**Tujuan penggunaan:** Mengevaluasi apakah kombinasi beberapa pohon keputusan dapat menghasilkan performa prediksi dan kemampuan generalisasi yang lebih baik dibandingkan model pohon tunggal.

### 3. Evaluasi Model

Evaluasi dilakukan menggunakan beberapa metrik klasifikasi untuk memperoleh gambaran performa model secara menyeluruh.

- **Accuracy:** Mengukur proporsi prediksi yang benar terhadap seluruh data evaluasi.
- **Precision:** Mengukur proporsi prediksi churn yang benar-benar merupakan pelanggan churn.
- **Recall:** Mengukur proporsi pelanggan churn aktual yang berhasil diidentifikasi oleh model.
- **F1-Score:** Mengukur keseimbangan antara precision dan recall melalui rata-rata harmonik keduanya.
- **ROC-AUC:** Mengukur kemampuan model dalam membedakan kelas churn dan non-churn pada berbagai ambang klasifikasi.

Recall menjadi salah satu metrik penting apabila tujuan utama sistem adalah meminimalkan jumlah pelanggan berisiko yang tidak terdeteksi. Namun, precision tetap perlu diperhatikan agar strategi retensi tidak diarahkan secara berlebihan kepada pelanggan yang sebenarnya tidak berisiko churn.

### 4. Interpretasi Model

Tahap interpretasi bertujuan memahami fitur-fitur yang berkontribusi terhadap prediksi model.

Teknik interpretasi yang dapat digunakan meliputi:

- **Feature Importance:** Mengidentifikasi tingkat kepentingan fitur pada model berbasis pohon.
- **Koefisien Model:** Menganalisis arah dan besarnya hubungan fitur dengan prediksi pada Logistic Regression.
- **Permutation Importance:** Mengevaluasi kontribusi fitur berdasarkan perubahan performa model ketika nilai fitur tersebut diacak.

Beberapa fitur yang dapat dianalisis lebih lanjut antara lain `tenure`, `Contract`, `MonthlyCharges`, dan `PaymentMethod`. Tingkat kepentingan setiap fitur ditentukan berdasarkan hasil eksperimen, bukan diasumsikan sejak awal.

## Tujuan Proyek

1. Mengeksplorasi data pelanggan untuk mengidentifikasi pola yang berkaitan dengan churn.
2. Membangun alur pra-pemrosesan data yang sesuai untuk permasalahan klasifikasi.
3. Melatih dan membandingkan lima algoritma Machine Learning.
4. Mengevaluasi performa model menggunakan berbagai metrik klasifikasi.
5. Menganalisis fitur yang berkontribusi terhadap hasil prediksi.
6. Mengeksplorasi pemanfaatan hasil prediksi untuk mendukung strategi retensi pelanggan.

## Hasil yang Diharapkan

Proyek ini diharapkan menghasilkan analisis komparatif terhadap lima algoritma klasifikasi yang diuji serta mengidentifikasi model dengan performa yang sesuai berdasarkan hasil evaluasi.

Analisis fitur juga diharapkan memberikan wawasan mengenai karakteristik pelanggan yang berkaitan dengan kemungkinan churn. Informasi tersebut dapat menjadi dasar untuk menyusun strategi retensi yang lebih terarah dan berbasis data.

Model terbaik, nilai evaluasi, serta fitur yang paling berpengaruh akan ditentukan dan didokumentasikan berdasarkan hasil eksperimen yang sebenarnya.

## Batasan Penggunaan

Proyek ini dikembangkan untuk keperluan pembelajaran, eksperimen, dan analisis data. Hasil prediksi merupakan estimasi berdasarkan pola historis dalam dataset, bukan kepastian bahwa seorang pelanggan akan berhenti berlangganan.

Penggunaan model pada lingkungan bisnis nyata memerlukan validasi lebih lanjut terhadap kualitas data, performa pada data baru, serta kesesuaian hasil prediksi dengan kebutuhan operasional.
# 5. Pustaka (Libraries) Python yang Digunakan
Dalam membangun proyek Machine Learning ini, kita menggunakan beberapa pustaka standar industri (standard stack) di Python:

1. Pandas & NumPy

* Fungsi: Untuk manipulasi, pembersihan (data cleaning), dan pengolahan data tabular.

* Peran di Proyek: Membaca file CSV, menangani nilai kosong (missing values), mengubah tipe data, dan melakukan operasi matematika dasar.

2. Scikit-Learn (sklearn)

* Fungsi: Pustaka utama untuk Machine Learning di Python.

* Peran di Proyek:

* Menyediakan algoritma klasifikasi: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Decision Tree, dan Random Forest.

* Menyediakan alat pra-pemrosesan seperti StandardScaler (untuk normalisasi angka) dan OneHotEncoder / pd.get_dummies (untuk mengubah teks kategori menjadi angka).

* Menyediakan metrik evaluasi seperti confusion_matrix, classification_report, dan roc_auc_score.

3. Imbalanced-Learn (imbalanced-learn / imblearn)

* Fungsi: Pustaka khusus untuk menangani data yang tidak seimbang (imbalanced data).

* Peran di Proyek: Menyediakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan jumlah data kelas churn dan tidak churn pada data latih.

4. Matplotlib & Seaborn

* Fungsi: Pustaka utama untuk visualisasi data dan grafik di Python.

* Peran di Proyek: Membuat berbagai macam grafik analisis data dan evaluasi model.
