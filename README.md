🎙️ TIMIT Gerçek Zamanlı Cinsiyet Tanıma Sistemi

TIMIT veri seti üzerinde eğitilmiş ECAPA-TDNN mimarisini kullanan, gürültülü ortamlarda ve gerçek zamanlı (Real-Time) çalışabilen uçtan uca bir ses analiz sistemidir
🌟 Temel Özellikler

Canlı Mikrofon Analizi: Asenkron işleme (threading) sayesinde 200ms'nin altında gecikme ile anlık cinsiyet tahmini.

(ECAPA-TDNN):Channel Attention mekanizması ile gürültülü ortamlarda yüksek başarı.

Akış Simülasyonu (Senaryo 2): YouTube veya uzun ses kayıtları üzerinde konuşmacı değişimi (Speaker Diarization benzeri) ve zaman çizelgesi analizi.

Sessizlik Tespiti (VAD): Enerji tabanlı filtreleme ile sessiz anlarda işlemciyi yormaz ve hatalı tahminleri önler.

Modern Arayüz: Streamlit ile geliştirilmiş, parametreleri dinamik olarak değiştirilebilen profesyonel kontrol paneli.

🛠️ Teknoloji Stack

Yapay Zeka: PyTorch, Torchaudio

Model: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation)

Arayüz: Streamlit

Sinyal İşleme: NumPy, SciPy, SoundDevice

Veri Yönetimi: yt-dlp, FFmpeg

⚙️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1. Gerekli Kütüphaneleri Yükleyin

Tercihen temiz bir Python ortamında (Anaconda veya venv) çalışın.

pip install -r requirements.txt


2. FFmpeg Kurulumu

Ses işleme ve YouTube indirmeleri için sisteminizde FFmpeg yüklü olmalıdır.

Windows: ffmpeg.exe dosyasını indirin ve projenin ana klasörüne (app.py yanına) koyun.

Linux: sudo apt-get install ffmpeg

Mac: brew install ffmpeg

🚀 Kullanım

1. Uygulamayı Başlatma

Ana kontrol panelini açmak için terminale şu kodu yazın:

streamlit run app.py


Otomatik olarak tarayıcınızda http://localhost:8501 adresi açılacaktır.

2. Test Verisi Oluşturma (Senaryo 2 İçin)

Modelin konuşmacı değişimlerine (Erkek -> Kadın) tepkisini ölçmek için otomatik test verisi oluşturucu scripti çalıştırın. Bu script, Steve Jobs, Emma Watson gibi net sesleri indirip birleştirir.

python prepare_data.py


Bu işlem sonunda klasörünüzde scenario2_final.wav dosyası oluşacaktır.

Uygulamada "Senaryo 2: Akış Simülasyonu" sekmesine gelip bu dosyayı yükleyerek testi başlatabilirsiniz.

📂 Dosya Yapısı

TIMIT-Gender-Recognition/
│
├── app.py                  # Ana Uygulama Kodu (Frontend & Backend)
├── prepare_data.py         # Test Verisi Hazırlama Scripti (YouTube Downloader)
├── best_model_ecapa.pth    # Eğitilmiş Model Ağırlıkları
├── requirements.txt        # Gerekli Python Kütüphaneleri
├── ffmpeg.exe              # Ses İşleme Aracı (Windows için gereklidir)
└── README.md               # Proje Dokümantasyonu
