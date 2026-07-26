import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Klasifikasi Sampah YOLOv8", page_icon="♻️", layout="wide")

@st.cache_resource
def init_model():
    return YOLO('best.pt')

model = init_model()

# --- BAGIAN SIDEBAR (Menu Samping) ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    st.write("Silakan tentukan parameter lingkungan sebelum memulai identifikasi.")
    
    lokasi = st.radio(
        "Lingkungan pengambilan citra:", 
        ["Rumah Tangga / Dalam Ruangan", "TPA / Pengepul / Luar Ruangan"]
    )
    
    st.divider()


# --- BAGIAN UTAMA (Main Page) ---
st.title("♻️ Deteksi dan Klasifikasi Sampah Cerdas")
st.markdown("Aplikasi ini dirancang untuk mengidentifikasi kategori sampah secara otomatis menggunakan model YOLOv8. Silakan masukkan data citra melalui salah satu metode di bawah ini.")

tab1, tab2 = st.tabs(["📂 Unggah Citra", "📸 Kamera Perangkat"])

source_img = None

# TAB 1: Upload File
with tab1:
    st.info("Format berkas yang didukung: JPG, JPEG, dan PNG.")
    img_upload = st.file_uploader("Pilih berkas citra untuk dianalisis", type=["jpg", "jpeg", "png"])
    if img_upload: 
        source_img = img_upload

# TAB 2: Kamera Statis
with tab2:
    st.info("Berikan izin akses kamera pada peramban (browser) Anda, lalu centang kotak di bawah ini.")
    aktifkan_kamera = st.checkbox("Aktifkan Kamera Perangkat")
    if aktifkan_kamera:
        cam = st.camera_input("Ambil citra sampah secara langsung")
        if cam: 
            source_img = cam

st.divider()

# --- LOGIKA PROSES DETEKSI GAMBAR STATIS ---
if source_img is not None:
    col_kiri, col_kanan = st.columns(2, gap="large")
    
    img = Image.open(source_img)
    
    with col_kiri:
        st.markdown("### 📷 Pratinjau Citra Masukan")
        st.image(img, width='stretch')
    
    with col_kanan:
        st.markdown("### 🔍 Hasil Identifikasi")
        
        if st.button('Mulai Analisis Citra', type="primary", use_container_width=True):
            with st.spinner('Sistem sedang melakukan inferensi...'):
                res = model(img)
                
                # fix warna bgr ke rgb
                res_img = Image.fromarray(res[0].plot()[..., ::-1])
                st.image(res_img, width='stretch')
                
                # menampung nama class yang terdeteksi
                detected = []
                for b in res[0].boxes:
                    id_cls = int(b.cls[0])
                    detected.append(model.names[id_cls])
                
                # hapus nama class untuk yang ganda
                unik = list(set(detected))
                
                st.markdown("---")
                st.markdown("### 📋 Kesimpulan & Rekomendasi Penanganan")
                
                if len(unik) == 0:
                    st.warning("**Hasil:** Tidak ditemukan objek sampah yang terdefinisi pada citra masukan.")
                else:
                    # logic klasifikasi
                    if 'anorganik_basah' in unik and 'organik' in unik:
                        st.error("**Kategori: Sampah Campuran**")
                        st.write("**Tindakan:** Terdeteksi keberadaan material organik dan anorganik secara bersamaan. Wajib dilakukan pemilahan tahap awal secara manual sebelum diproses lebih lanjut.")
                    
                    elif 'anorganik_basah' in unik:
                        if lokasi == "TPA / Pengepul / Luar Ruangan":
                            st.warning("**Kategori: Anorganik (Kondisi Lapangan)**")
                            st.write("**Tindakan:** Sampah berupa tumpukan massal atau industri. Sistem merekomendasikan penyortiran lanjutan berdasarkan spesifikasi material.")
                        else:
                            st.error("**Kategori: Anorganik Terkontaminasi**")
                            st.write("**Tindakan:** Sampah anorganik terindikasi dalam kondisi kotor atau basah. Material harus melalui tahap pencucian terlebih dahulu guna mencegah kerusakan pada mesin daur ulang.")
                            
                    elif 'anorganik' in unik:
                        st.success("**Kategori: Anorganik Bersih**")
                        st.write("**Tindakan:** Material teridentifikasi dalam kondisi bersih dan utuh. Sampah siap didistribusikan ke tahap pencacahan industri.")
                        
                    elif 'organik' in unik:
                        st.success("**Kategori: Organik**")
                        st.write("**Tindakan:** Material organik siap diproses langsung untuk kebutuhan produksi kompos atau budidaya pakan maggot BSF (Black Soldier Fly).")