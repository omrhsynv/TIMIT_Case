import os
import torch
import torchaudio
import yt_dlp
import numpy as np

# --- AYARLAR ---
OUTPUT_FILENAME = "scenario2_final.wav"
SAMPLE_RATE = 16000
CLIP_DURATION = 10 # Her konuşmacıdan 10 saniye 
SILENCE_DURATION = 3 # Araya 3 saniye sessizlik koy

# --- SEÇİLEN VİDEOLAR ---
URLS = [
    # (URL, Etiket, Açıklama)
    ("https://www.youtube.com/watch?v=UF8uR6Z6KLc", "M", "Steve Jobs (Stanford Speech)"),
    ("https://www.youtube.com/watch?v=gkjW9PZBRfk", "F", "Emma Watson (UN Speech)"),
    ("https://www.youtube.com/watch?v=TVsounscj4U", "M", "MKBHD (Tech Review)"),
    ("https://www.youtube.com/watch?v=2AKX0dGq_x8", "F", "Oprah Winfrey (Interview)")
]

def download_and_process(url, label, desc):
    print(f"⬇İndiriliyor: {desc}...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav','preferredquality': '192'}],
        'outtmpl': 'temp_dl',
        'quiet': True,
        'noplaylist': True
    }
    
    try:
        if os.path.exists("temp_dl.wav"): os.remove("temp_dl.wav")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Sesi Yükle
        sig, sr = torchaudio.load("temp_dl.wav")
        
        # Resample (16kHz yap)
        if sr != SAMPLE_RATE:
            sig = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(sig)
        
        # Mono Yap (Stereo ise)
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)
            
        # (Videonun başındaki introyu atlamak için %20 ileriden başla)
        start_offset = int(sig.shape[1] * 0.2)
        target_len = SAMPLE_RATE * CLIP_DURATION
        
        if sig.shape[1] > start_offset + target_len:
            sig = sig[:, start_offset : start_offset + target_len]
        else:
            # Ses kısaysa başa sar
            padding = target_len - sig.shape[1]
            sig = torch.nn.functional.pad(sig, (0, padding))
            
        return sig
    except Exception as e:
        print(f"Hata ({desc}): {e}")
        return None

def create_silence(duration):
    return torch.zeros(1, SAMPLE_RATE * duration)

def main():
    final_audio = []
    print(f"Senaryo 2 Dosyası Hazırlanıyor... (Toplam {len(URLS)} konuşmacı)")
    print("-" * 50)
    
    # Başlangıça biraz sessizlik koy (Grafik otursun diye)
    final_audio.append(create_silence(2))
    
    for url, label, desc in URLS:
        clip = download_and_process(url, label, desc)
        if clip is not None:
            final_audio.append(clip)
            print(f"Eklendi: {desc}")
            
            # Araya sessizlik ekle
            final_audio.append(create_silence(SILENCE_DURATION))
    
    if final_audio:
        # Hepsini Birleştir
        full_tensor = torch.cat(final_audio, dim=1)
        
        # Kaydet
        torchaudio.save(OUTPUT_FILENAME, full_tensor, SAMPLE_RATE)
        print("-" * 50)
        print(f"İŞLEM TAMAMLANDI!")
        print(f" Oluşturulan Dosya: {OUTPUT_FILENAME}")
        print(f"⏱Toplam Süre: {full_tensor.shape[1] / SAMPLE_RATE:.2f} saniye")
        print("Şimdi 'app.py'yi çalıştırıp Senaryo 2 sekmesine bu dosyayı yükleyin.")
    else:
        print("Hiçbir ses indirilemedi.")

if __name__ == "__main__":
    main()