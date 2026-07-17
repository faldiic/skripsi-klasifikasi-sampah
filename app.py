import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Klasifikasi Sampah", layout="centered")

# cache model biar ga berat pas di-refresh
@st.cache_resource
def init_model():
    return YOLO('best.pt')

model = init_model()

st.title("Sistem Klasifikasi Sampah YOLOv8")
st.markdown("Silakan unggah foto atau gunakan kamera untuk proses analisis sistem.")

lokasi = st.radio(
    "Pilih lokasi pengambilan foto:", 
    ["Rumah Tangga / Indoor", "TPA / Pengepul / Outdoor"]
)

# bikin tabs buat pilihan metode input
tab1, tab2 = st.tabs(["Unggah Gambar", "Gunakan Kamera"])

source_img = None

with tab1:
    img_upload = st.file_uploader("Pilih file gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
    if img_upload: 
        source_img = img_upload

with tab2:
    # kasih checkbox biar kamera bisa dimatiin buat hemat resource
    st.info("Centang kotak di bawah untuk mengaktifkan kamera. Hilangkan centang jika sudah tidak digunakan.")
    aktifkan_kamera = st.checkbox("Aktifkan Kamera")
    
    if aktifkan_kamera:
        cam = st.camera_input("Ambil gambar dari perangkat")
        if cam: 
            source_img = cam

# proses deteksi kalau gambar udah masuk
if source_img is not None:
    img = Image.open(source_img)
    st.image(img, caption='Pratinjau Gambar', width='stretch')
    
    if st.button('Mulai Deteksi'):
        with st.spinner('Sedang memproses gambar...'):
            res = model(img)
            
            # benerin warna bgr ke rgb bawaan opencv
            res_img = Image.fromarray(res[0].plot()[..., ::-1])
            
            st.subheader("Hasil Deteksi")
            st.image(res_img, width='stretch')
            
            # tampung semua class yg kedeteksi
            detected = []
            for b in res[0].boxes:
                id_cls = int(b.cls[0])
                detected.append(model.names[id_cls])
            
            # hapus class yg duplikat biar rapi
            unik = list(set(detected))
            # print("deteksi:", unik) # buat ngetes doang pas run lokal
            
            st.markdown("### Detail Analisis")
            
            if len(unik) == 0:
                st.warning("Tidak ada objek sampah yang terdeteksi dengan jelas pada gambar.")
            else:
                # logic rule-based buat rekomendasi
                if 'anorganik_basah' in unik and 'organik' in unik:
                    st.error("Kategori: Sampah Campuran")
                    st.write("Catatan: Terdapat campuran material organik dan anorganik. Lakukan pemilahan manual sebelum memproses lebih lanjut.")
                
                elif 'anorganik_basah' in unik:
                    if lokasi == "TPA / Pengepul / Outdoor":
                        st.warning("Kategori: Anorganik Kondisi Lapangan")
                        st.write("Catatan: Sampah berupa tumpukan massal. Diperlukan penyortiran lanjutan berdasarkan jenis material.")
                    else:
                        st.error("Kategori: Anorganik Terkontaminasi")
                        st.write("Catatan: Sampah anorganik dalam kondisi kotor atau basah. Wajib dicuci terlebih dahulu agar tidak merusak mesin daur ulang.")
                        
                elif 'anorganik' in unik:
                    st.success("Kategori: Anorganik Bersih")
                    st.write("Catatan: Material dalam kondisi bersih dan dapat langsung diproses ke tahap pencacahan.")
                    
                elif 'organik' in unik:
                    st.success("Kategori: Organik")
                    st.write("Catatan: Material organik siap diproses untuk kebutuhan pembuatan kompos atau pakan maggot BSF.")