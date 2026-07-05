import torch
from PIL import Image, ImageDraw
from transformers import pipeline
import numpy as np

def create_synthetic_scene():
    """
    Derinlik tahminini test etmek amacıyla yapay bir 3D sahne oluşturur.
    Bu sayede harici bir görsel indirmeye gerek kalmadan kod doğrudan çalıştırılabilir.
    """
    # 512x512 boyutlarında gökyüzü mavisi arka plana sahip bir görsel oluştur
    img = Image.new("RGB", (512, 512), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    
    # Perspektif hissi vermek için yeşil bir zemin (yer) çiz
    draw.polygon([(0, 300), (512, 300), (512, 512), (0, 512)], fill=(34, 139, 34))
    
    # Ön planda duran kırmızı bir küre (daire)
    draw.ellipse([120, 220, 280, 380], fill=(220, 20, 60))
    
    # Arka planda duran daha küçük mavi bir kutu (dikdörtgen)
    draw.rectangle([340, 240, 420, 320], fill=(30, 144, 255))
    
    return img

def main():
    print("1. Yapay 3D sahne oluşturuluyor...")
    image = create_synthetic_scene()
    image.save("giris_sahnesi.png")
    print("-> Giriş görseli 'giris_sahnesi.png' olarak kaydedildi.")

    print("\n2. Depth Anything V2 modeli Hugging Face üzerinden yükleniyor...")
    # Hızlı ve hafif çalışması için Depth Anything V2 Small modelini tercih ediyoruz
    try:
        depth_estimator = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Small-hf"
        )
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        print("Lütfen 'transformers' ve 'torch' kütüphanelerinin güncel olduğundan emin olun.")
        return

    print("\n3. Tek gözlü derinlik tahmini (Monocular Depth Estimation) gerçekleştiriliyor...")
    # Model görseli işler ve derinlik haritasını (depth map) üretir
    result = depth_estimator(image)
    
    # Çıktı olarak gelen derinlik haritası bir PIL görselidir
    depth_map = result["depth"]
    depth_map.save("derinlik_haritasi.png")
    print("-> Derinlik haritası 'derinlik_haritasi.png' olarak kaydedildi.")

    print("\n4. Karşılaştırma görseli hazırlanıyor...")
    # Orijinal görsel ile derinlik haritasını yan yana birleştir
    comparison = Image.new("RGB", (1024, 512))
    comparison.paste(image, (0, 0))
    # Derinlik haritasını RGB formatına dönüştürerek sağ tarafa yapıştır
    comparison.paste(depth_map.convert("RGB"), (512, 0))
    comparison.save("karsilastirma_sonucu.png")
    print("-> Yan yana karşılaştırma görseli 'karsilastirma_sonucu.png' olarak kaydedildi.")
    print("\nİşlem başarıyla tamamlandı! Çıktıları klasörünüzde görebilirsiniz.")

if __name__ == "__main__":
    main()