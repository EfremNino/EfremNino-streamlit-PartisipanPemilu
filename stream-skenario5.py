import pickle
import streamlit as st
import numpy as np

# Membaca model dan scaler dari file pickle
with open('skenario5new_model.sav', 'rb') as file:
    partisipasi_model = pickle.load(file)
with open('skenario5new_scaler.sav', 'rb') as file:
    scaler = pickle.load(file)

# Judul web
st.title('🗳️ Prediksi Pola Partisipasi Pemilu')

# Deskripsi aplikasi
st.write("""
    Selamat datang di aplikasi **Prediksi Pola Partisipasi Pemilu Di Kota Kupang**! 🎉
    Aplikasi ini memprediksi pola partisipasi pemilu di Kota Kupang berdasarkan beberapa variabel seperti jenis kelamin, umur, pendidikan, pekerjaan, 
    keterlibatan kampanye, keanggotaan partai politik, dan penggunaan hak pilih.
""")

# Input dari pengguna dengan sidebar
st.sidebar.header('📝 Input Data Partisipasi')
st.sidebar.write("Silakan masukkan data untuk memprediksi pola partisipasi:")

jenis_kelamin = st.sidebar.selectbox('Pilih Jenis Kelamin', ['Laki-Laki', 'Perempuan'])
umur = st.sidebar.selectbox('Pilih Kategori Umur', ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-55'])
pendidikan = st.sidebar.selectbox('Pilih Tingkat Pendidikan', ['SMP', 'SMA', 'S1', 'S2', 'S3'])
pekerjaan = st.sidebar.selectbox('Pilih Pekerjaan', ['Mahasiswa', 'Petani', 'Wiraswasta', 'Guru/Dosen/PNS'])
terlibat_kampanye = st.sidebar.selectbox('Apakah Terlibat Kampanye?', ['Tidak Pernah', 'Pernah'])
anggota_parpol = st.sidebar.selectbox('Apakah Anggota Partai Politik?', ['Tidak', 'Ya'])
menggunakan_hak_pilih = st.sidebar.selectbox('Apakah Menggunakan Hak Pilih?', ['Tidak', 'Ya'])

# Mapping fitur sesuai dengan deskripsi yang diberikan
mapping_jenis_kelamin = {'Laki-Laki': 1, 'Perempuan': 0.5}
mapping_umur = {'18-24': 0.4, '25-29': 0.6, '30-34': 0.7, '35-39': 0.8, '40-44': 1, '45-49': 0.9, '50-55': 0.5}
mapping_pendidikan = {'SMP': 0.3, 'SMA': 0.4, 'S1': 0.8, 'S2': 0.9, 'S3': 1}
mapping_pekerjaan = {'Mahasiswa': 0.1, 'Petani': 0.2, 'Wiraswasta': 0.3, 'Guru/Dosen/PNS': 0.5}
mapping_kampanye = {'Tidak Pernah': 0.5, 'Pernah': 1}
mapping_parpol = {'Tidak': 0.5, 'Ya': 1}
mapping_hak_pilih = {'Tidak': 0.5, 'Ya': 1}

# Menambahkan tombol prediksi
if st.sidebar.button('🔍 Prediksi'):
    # Memastikan semua input valid dan melakukan mapping
    try:
        input_data = (
            mapping_jenis_kelamin[jenis_kelamin],
            mapping_umur[umur],
            mapping_pendidikan[pendidikan],
            mapping_pekerjaan[pekerjaan],
            mapping_kampanye[terlibat_kampanye],
            mapping_parpol[anggota_parpol],
            mapping_hak_pilih[menggunakan_hak_pilih]
        )
    except KeyError:
        st.error("Pastikan semua input valid dan diisi dengan pilihan yang tersedia.")
        st.stop()

    input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

    # Preprocessing data baru
    std_data = scaler.transform(input_data_as_numpy_array)

    # Debug: Menampilkan data yang akan diprediksi
    st.write("### Data setelah preprocessing:")
    st.dataframe(std_data)

    # Melakukan prediksi
    parti_prediction = partisipasi_model.predict(std_data)

    # Menampilkan hasil prediksi
    if parti_prediction[0] <= 0.4:
        parti_diagnosis = 'Pola Partisipasi Pasif'
    else:
        parti_diagnosis = 'Pola Partisipasi Aktif'

    st.success(f"Prediksi: {parti_diagnosis}")

# Menambahkan informasi tambahan atau visualisasi jika diperlukan
st.write("""
    ### Informasi Tambahan
    Berikut adalah kode-kode variabel sesuai dengan kategori:
    - **Jenis Kelamin**: 1 untuk Laki-laki, 0.5 untuk Perempuan
    - **Umur**: 
        - 18-24: 0.4 
        - 25-29: 0.6 
        - 30-34: 0.7 
        - 35-39: 0.8 
        - 40-44: 1 
        - 45-49: 0.9 
        - 50-55: 0.5
    - **Pendidikan**: 
        - SMP: 0.3 
        - SMA: 0.4 
        - S1: 0.8 
        - S2: 0.9 
        - S3: 1
    - **Pekerjaan**: 
        - Mahasiswa: 0.1 
        - Petani: 0.2 
        - Wiraswasta: 0.3 
        - Guru/Dosen/PNS: 0.5
    - **Keterlibatan Kampanye**: 0.5 untuk Tidak Pernah, 1 untuk Pernah
    - **Keanggotaan Partai Politik**: 0.5 untuk Tidak, 1 untuk Ya
    - **Menggunakan Hak Pilih**: 0.5 untuk Tidak, 1 untuk Ya
""")
