# Genetik Algoritma Destekli Özellik Seçimi ve XAI ile Model Yorumlama — PM2.5 Hava Kirliliği Tahmini

Bu proje, saatlik PM2.5 hava kirliliği değerinin tahmin edilmesi problemini ele almaktadır. Çalışmada önce **Genetik Algoritma** ile en bilgilendirici özellikler seçilmiş, ardından üç farklı makine öğrenmesi modeli hiperparametre optimizasyonu ile eğitilmiş, en başarılı model ise **Açıklanabilir Yapay Zekâ (XAI)** yöntemleriyle yorumlanmıştır.

## İçerik

- `optimizasyon_proje2.py` — Tüm analiz adımlarını içeren Python kodu (veri üretimi, GA ile özellik seçimi, model optimizasyonu, XAI analizleri, grafikler)
- `Optimizasyon_Teknikleri_Proje.pdf` — Çalışmanın detaylı raporu

## Veri Seti Hakkında Önemli Not

Rapor, veri setinin istatistiksel yapısının **UCI Machine Learning Repository — Beijing PM2.5 Data Set** referans alınarak oluşturulduğunu belirtmektedir. Kod içinde kullanılan veri, gerçek UCI veri setinin doğrudan indirilmiş bir kopyası **değildir**; gerçek veri setinin genel istatistiksel örüntülerine (günlük/haftalık/aylık döngüler, değişkenler arası ilişkiler) benzer şekilde **sentetik olarak üretilmiştir**. Bu yaklaşım, kodun harici bir veri dosyasına ihtiyaç duymadan bağımsız şekilde çalışabilmesini sağlamak amacıyla tercih edilmiştir. Gerçek UCI veri setine şu adresten ulaşılabilir: https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data

## Yöntem Özeti

1. **Genetik Algoritma ile Özellik Seçimi:** 8 değişkenden 3'ü (humidity, hour, pm25_roll6) seçilmiş, 5'i elenmiştir.
2. **Model Optimizasyonu:**
   - Random Forest — Grid Search
   - Gradient Boosting — Random Search
   - SVR — Grid Search
3. **XAI Analizleri:** Permutation Feature Importance ve LIME ile en başarılı modelin (SVR) karar süreci yorumlanmıştır.

## Sonuçlar

| Model | Optimizasyon | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|---|
| Random Forest | Grid Search | 4.8749 | 3.7623 | 0.9368 | %6.35 |
| Gradient Boosting | Random Search | 4.6635 | 3.5812 | 0.9422 | %5.88 |
| **SVR** | **Grid Search** | **4.4029** | **3.4107** | **0.9485** | **%5.59** |

En iyi performansı SVR modeli göstermiştir.

## Kullanılan Kütüphaneler

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

## Çalıştırma

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
python optimizasyon_proje2.py
```

## Kaynaklar

Raporun kaynakça bölümünde (Bölüm 10) tüm akademik referanslar listelenmiştir.

## Yazar

Aslınur Benli — 20231101045
