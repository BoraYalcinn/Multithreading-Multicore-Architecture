# Lab-N — <Başlık>

**Ad Soyad:** Boray Alçıntır
**Tarih:** YYYY-MM-DD
**Kod:** [`../../Lab-N/`](../../Lab-N/)

## Test Ortamı

| | |
|---|---|
| CPU | örn. AMD Ryzen 7 5800H |
| Fiziksel / mantıksal çekirdek | 8 / 16 |
| İşletim sistemi | Ubuntu 24.04, kernel 6.x |
| Derleyici | g++ 13.2 |
| Derleme bayrakları | `-std=c++20 -O2 -pthread` |
| Tekrar sayısı | 3 (ortalama raporlandı) |

> `lscpu | head -20` ve `g++ --version` çıktılarıyla doldur.

---

## 1. Amaç

Bu labda neyin ölçüldüğü / gösterildiği, 2–3 cümle.

## 2. Yöntem

Kurulan deney: hangi iş yükü, hangi parametreler (thread/process sayısı, problem
boyutu), ne ölçüldü.

## 3. Sonuçlar

| p (worker) | Tp (s) | Speedup Sp = T1/Tp | Efficiency Ep = Sp/p |
|---|---|---|---|
| 1 | | 1.00 | 1.00 |
| 2 | | | |
| 4 | | | |
| 8 | | | |

![Speedup ve efficiency eğrileri](assets/<grafik>.png)

## 4. Yorum

- Ölçülen speedup ideal `Sp = p` doğrusundan nerede ve neden sapıyor?
- Amdahl yasası açısından seri kalan kısım ne kadar?
- Çekirdek sayısını aştıktan sonra ne oluyor (oversubscription, context switch)?
- CPU-bound / I/O-bound ayrımı sonucu nasıl değiştiriyor?

## 5. Karşılaşılan Sorunlar

Race condition, deadlock, false sharing vb. gözlemler ve nasıl çözüldüğü.

## 6. Sonuç

Çıkarılan ana dersler, 3–5 madde.
