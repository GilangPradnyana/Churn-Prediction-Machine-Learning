# 1. Apa itu Customer Churn Prediction?
Customer Churn Prediction adalah sistem berbasis Machine Learning dan data analytics untuk mengidentifikasi probabilitas pelanggan berhenti berlangganan (churn) maupun tetap bertahan. Melalui pendekatan ini, bisnis dapat mendeteksi indikasi awal hilangnya pelanggan secara proaktif sebelum benar-benar terjadi.

# 2. Masalah Apa yang Diselesaikan?
Masalah utama yang diselesaikan dalam proyek ini adalah tingginya tingkat perpindahan pelanggan ke kompetitor, di mana perusahaan sering kali terlambat menyadarinya (bersifat reaktif).

Secara rinci, masalah bisnis yang diselesaikan meliputi:

Kehilangan Pendapatan Berulang (Recurring Revenue): Setiap pelanggan yang pergi mengurangi pemasukan bulanan stabil perusahaan.

Biaya Akuisisi yang Mahal: Mencari pelanggan baru terbukti jauh lebih mahal (bisa 5–7 kali lipat) dibandingkan biaya untuk mempertahankan pelanggan yang sudah ada. Oleh karena itu, membiarkan pelanggan lama pergi adalah pemborosan.

Kurangnya Deteksi Dini: Tanpa sistem AI, perusahaan biasanya baru tahu pelanggan marah setelah mereka menutup akun. Dengan prediksi ini, perusahaan bisa bertindak proaktif (misalnya memberikan diskon atau layanan khusus) kepada pelanggan yang berisiko tinggi.

# 3. Data yang Digunakan 
Dataset yang dipakai adalah Telco Customer Churn dari Kaggle, dengan spesifikasi sebagai berikut:

Ukuran Data: Terdiri dari 7.043 baris data pelanggan dan 21 kolom (fitur).

Kolom Target (Label): Kolom Churn (bernilai Yes atau No), yang menunjukkan apakah pelanggan tersebut berhenti berlangganan.

Kategori Fitur dalam Dataset Anda:

Demografi Pelanggan: gender, SeniorCitizen (lansia), Partner (memiliki pasangan), Dependents (tanggungan).

Layanan yang Diambil: PhoneService, MultipleLines, InternetService (DSL/Fiber optic), OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.

Informasi Akun & Pembayaran:

tenure: Berapa bulan pelanggan telah berlangganan.

Contract: Jenis kontrak (Bulanan / Month-to-month, 1 tahun, 2 tahun).

PaperlessBilling: Tagihan tanpa kertas (Ya/Tidak).

PaymentMethod: Metode pembayaran (contoh: Electronic check, Mailed check, dll).

MonthlyCharges: Besaran tagihan setiap bulan.

TotalCharges: Akumulasi total tagihan selama berlangganan.

# 4. Metode yang Dipakai
Proyek ini menggunakan pendekatan Supervised Learning (karena data target Churn Yes/No sudah tersedia). Tahapan metode yang digunakan meliputi:

## Pra-pemrosesan Data (Data Preprocessing & Cleaning):

Mengatasi data kosong atau tipe data yang tidak sesuai (misalnya mengubah kolom TotalCharges dari teks/object menjadi numerik).

Encoding: Mengubah data teks/kategori (seperti jenis kelamin, jenis kontrak, metode pembayaran) menjadi format angka menggunakan One-Hot Encoding.

Scaling: Menyamakan rentang nilai pada fitur numerik (tenure, MonthlyCharges) menggunakan StandardScaler.

Handling Imbalanced Data: Menggunakan teknik seperti SMOTE jika jumlah pelanggan yang churn jauh lebih sedikit dibanding yang bertahan, agar model AI tidak berat sebelah.

## Pemodelan (Machine Learning Algorithms):
Dalam proyek ini, Anda akan membandingkan 5 algoritma klasifikasi berikut:
* Logistic Regression (Regresi Logistik):Fungsi: Model linier sederhana yang menghitung peluang (probability) keanggotaan kelas menggunakan fungsi sigmoid.Peran di Proyek: Digunakan sebagai model baseline (acuan dasar) untuk melihat performa awal sebelum mencoba model yang lebih kompleks.
* K-Nearest Neighbors (KNN):Fungsi: Mengklasifikasikan pelanggan baru berdasarkan kemiripan jarak (misal: Euclidean Distance) dengan $K$ pelanggan terdekat di dataset.Peran di Proyek: Menangkap kelompok pelanggan yang memiliki pola pembayaran atau karakteristik layanan yang hampir identik.
* Support Vector Machine / Classifier (SVC):Fungsi: Mencari bidang pemisah (hyperplane) optimal dengan margin terbesar yang memisahkan pelanggan churn dan tidak churn.Peran di Proyek: Sangat efektif untuk ruang data berdimensi tinggi setelah dilakukan One-Hot Encoding.
* Decision Tree (Pohon Keputusan):Fungsi: Membagi data secara hierarkis menggunakan aturan keputusan (if-else) berdasarkan metrik seperti Gini Impurity atau Entropy.Peran di Proyek: Memberikan alur logika yang mudah divisualisasikan dan dipahami oleh pihak manajemen bisnis.
* Random Forest Classifier:Fungsi: Metode Ensemble (Bagging) yang menggabungkan puluhan hingga ratusan pohon keputusan (Decision Trees) acak untuk menghasilkan satu prediksi akhir berbasis suara terbanyak (majority voting).Peran di Proyek: Menjadi model andalan (benchmark utama) karena sangat tahan terhadap overfitting dan memiliki akurasi tinggi pada data tabular.

## Evaluasi & Interpretasi:

Mengukur keberhasilan model menggunakan metrik seperti Recall, Precision, dan ROC-AUC (di mana Recall penting agar sedikit mungkin pelanggan churn yang lolos dari deteksi).

Melihat Feature Importance untuk mengetahui fitur mana yang paling kuat memengaruhi pelanggan untuk churn (misalnya: durasi tenure yang singkat dan jenis kontrak month-to-month).

# 5. Pustaka (Libraries) Python yang Digunakan
Dalam membangun proyek Machine Learning ini, kita menggunakan beberapa pustaka standar industri (standard stack) di Python:

# Pandas & NumPy

Fungsi: Untuk manipulasi, pembersihan (data cleaning), dan pengolahan data tabular.

Peran di Proyek:
* Membaca file CSV, menangani nilai kosong (missing values), mengubah tipe data, dan melakukan operasi matematika dasar.

# Scikit-Learn (sklearn)

Fungsi: Pustaka utama untuk Machine Learning di Python.

Peran di Proyek:

* Menyediakan algoritma klasifikasi: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Decision Tree, dan Random Forest.

* Menyediakan alat pra-pemrosesan seperti StandardScaler (untuk normalisasi angka) dan OneHotEncoder / pd.get_dummies (untuk mengubah teks kategori menjadi angka).

* Menyediakan metrik evaluasi seperti confusion_matrix, classification_report, dan roc_auc_score.

# Imbalanced-Learn (imbalanced-learn / imblearn)

Fungsi: Pustaka khusus untuk menangani data yang tidak seimbang (imbalanced data).

Peran di Proyek:
* Menyediakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan jumlah data kelas churn dan tidak churn pada data latih.

# Matplotlib & Seaborn

Fungsi: Pustaka utama untuk visualisasi data dan grafik di Python.

Peran di Proyek: 
* Membuat berbagai macam grafik analisis data dan evaluasi model.
