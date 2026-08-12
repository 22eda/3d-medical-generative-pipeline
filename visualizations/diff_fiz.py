import napari
import nibabel as nib
import numpy as np
import os

# --- AYARLAR ---
# Modelin ürettiği dosya (Az önce indirdiğimiz)
tahmin_dosya_yolu = 'predicted_ctn.nii.gz' 

# !!! DİKKAT: Buraya karşılaştırmak istediğin GERÇEK NIfTI dosyasının yolunu yazmalısın !!!
# Eğer bu dosya Athena sunucusundaysa, onu da indirmen gerekir.
gercek_veri_yolu = 'oldest_0002.nii.gz' 
# ----------------

def visualize_difference(pred_path, gt_path):
    if not os.path.exists(pred_path):
        print(f"Hata: Tahmin dosyası bulunamadı: {pred_path}")
        return
    if not os.path.exists(gt_path):
        print(f"Hata: Gerçek veri dosyası bulunamadı: {gt_path}")
        print("Lütfen kod içindeki 'gercek_veri_yolu'nu güncelleyin.")
        return

    print("Dosyalar yükleniyor...")
    # NIfTI dosyalarını yükle ve numpy dizisine çevir
    pred_img = nib.load(pred_path)
    pred_data = pred_img.get_fdata()
    
    gt_img = nib.load(gt_path)
    gt_data = gt_img.get_fdata()

    # Boyut kontrolü (İki görüntü aynı boyutta olmalı)
    if pred_data.shape != gt_data.shape:
        print(f"Hata: Boyut uyuşmazlığı!\nTahmin boyutu: {pred_data.shape}\nGerçek veri boyutu: {gt_data.shape}")
        # Görüntüleri en küçük ortak boyuta kırpma veya yeniden örnekleme (resampling) eklenebilir, 
        # ancak şimdilik hata verip çıkıyoruz.
        return

    # --- Fark Haritasını Hesapla ---
    # Mutlak farkı hesapla: |Tahmin - Gerçek|
    diff_map = np.abs(pred_data - gt_data)

    # --- Görselleştirme ---
    print("Napari başlatılıyor...")
    viewer = napari.Viewer(title=f'Generative CT Analizi: Fark Haritası')

    # 1. Katman: Gerçek Veri (Arka plan için gri tonlarında)
    layer_gt = viewer.add_image(gt_data, name='Gerçek CTn (GT)', colormap='gray', blending='additive', opacity=0.7)
    
    # 2. Katman: Model Tahmini (Üstüne bindir, yeşil tonlarında)
    layer_pred = viewer.add_image(pred_data, name='Tahmini CTn', colormap='green', blending='additive', opacity=0.5)

    # 3. Katman: Fark Haritası (En üste koy, ısı haritası ile)
    # Renk skalası: Düşük fark mavi, Yüksek fark kırmızıya döner.
    # 'magma' veya 'inferno' ısı haritası için iyi seçeneklerdir.
    layer_diff = viewer.add_image(diff_map, name='Mutlak Fark Haritası (|Pred - GT|)', colormap='magma', blending='additive', visible=True)
    
    # Bilgi notu
    print("\n--- GÖRSELLEŞTİRME KONTROLLERİ ---")
    print("Sağ paneldeki katman listesinden (Layers) katmanların görünürlüğünü açıp kapatabilirsin.")
    print("Fark Haritası (Difference Map) için 'magma' colormap'i kullanıldı.")
    print("Farkın yoğun olduğu kırmızı bölgeleri incele.")

    napari.run()

if __name__ == "__main__":
    # Kendi bilgisayarının terminalinden bu betiği çalıştırdığında bu blok çalışacak.
    # Windows için dosya yollarını çift ters eğik çizgi (\\) veya tek düz eğik çizgi (/) ile yazdığından emin ol.
    # Örn: tahmin_dosya_yolu = r'C:\Users\asus\Documents\predicted_ctn.nii.gz'
    
    # Eğer NIfTI dosyaları bu python dosyasıyla aynı klasörde değilse, tam yollarını buraya yapıştır.
    visualize_difference(tahmin_dosya_yolu, gercek_veri_yolu)