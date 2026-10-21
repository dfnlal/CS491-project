# CS491 – Fotoğraftan Düzenlenebilir 3D Model Çıkarma

Chen ve arkadaşlarının **3-Sweep** yönteminin (SIGGRAPH Asia 2013) temel sürümü.
Kullanıcı fotoğraftaki nesnenin her parçası için üç çizgi çiziyor, program
bu çizgilerden parçayı 3D'ye çeviriyor. Sonuç tek parça bir mesh değil,
ayrı ayrı düzenlenebilen parçalardan oluşan bir model.

## Kurulum

Python 3.11 ile 3.13 arası bir sürüm gerekiyor (3.13 önerilir).

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

İstersen fotoğrafı doğrudan da verebilirsin: `python main.py samples/synthetic_cylinder.png`.
`samples/` klasöründe denemek için yapay fotoğraflar var. Yeniden üretmek için:
`python tools/make_samples.py`.

Testler:

```bash
python -m pytest
```

## Kullanım: bir parçayı modellemek

Araç çubuğundan **Cylinder** (silindir) ya da **Cuboid** (kutu) seç, sonra:

1. **Basılı tutup sürükle:** ilk profil çizgisi.
   - Silindir: parçanın bir ucundaki genişliği (elipsin büyük ekseni).
   - Kutu: tabanın bir kenarı.
2. **Fareyi oynat ve tıkla:** ikinci profil çizgisi.
   - Silindir: elipsin ne kadar açık olduğu (küçük eksenin ucu).
   - Kutu: tabanın ikinci kenarı.
3. **Basılı tutup parça boyunca sürükle:** eksen. Sürüklerken parça fotoğraftaki
   kenarlara yapışıyor (önizleme beyaz, bulunan kenar noktaları pembe).
   Bırakınca parça eklenir.

`Esc` çizimi iptal eder, `Ctrl+Z` son parçayı siler. Fare tekerleği yakınlaştırır,
sağ ya da orta tuşla sürükleyince görüntü kayar.

Araç çubuğundaki seçenekler (makaledeki kısıtlar):

| Seçenek | Ne yapar |
|---|---|
| Straight axis | Ekseni başlangıçtan bitişe düz çizgi yapar |
| Size along axis: Constant / Linear | Kesitin boyutunu sabit ya da doğrusal değişen tutar. Kenarlar zayıfsa çok işe yarar |
| Flip depth | Parça kameraya doğru değil de ters yöne eğik çıktıysa diğer çözümü seçer |
| Show edges / Edge sensitivity | Bulunan kenarları gösterir, hassasiyeti ayarlar |

### 3D görünüm ve düzenleme

**Show 3D** (`Ctrl+3`) modeli ayrı bir pencerede gösterir. Listeden bir parça seçip
taşıyabilir, döndürebilir ya da ölçekleyebilirsin. Diğer parçalar değişmez; örneğin
masa tablasını büyütünce bacaklar olduğu gibi kalır. Seçili parçanın ekseni sarı çizilir.
Doku, parçanın fotoğraftaki yerinden alındığı için düzenlemeden sonra da parçanın üstünde kalır.

### Kaydetme ve dışa aktarma

- **File > Save project** (`.json`): çizgileri ve düzenlemeleri saklar, sonra açıp devam edebilirsin.
- **Export** (`.glb` ya da `.obj`): Godot, Unity ve Blender için. Her parça dosyada ayrı bir
  nesne olarak kalır. Model Y yukarı olacak şekilde, 1×1×1 kutuya sığacak boyutta ve
  zemine (Y = 0) oturmuş olarak kaydedilir. Fotoğraf doku olarak eklenir.

## Kod yapısı

| Dosya | İçerik | Makalede |
|---|---|---|
| `threesweep/geometry.py` | Koordinatlar, kamera, vektör yardımcıları | – |
| `threesweep/edges.py` | Fotoğrafı açma, Canny kenarları, kenara yapışma (ray casting) | Bölüm 3–4 |
| `threesweep/primitives.py` | İlk iki çizgiden 3D daire ya da dikdörtgen profil | Bölüm 4 "Profile", Ek A |
| `threesweep/sweep.py` | Profili eksen boyunca taşıma ve kenarlara yapıştırma, boyut kısıtları | Bölüm 4 "Sweeping" |
| `threesweep/mesh.py` | Parçalardan kapalı mesh, fotoğraftan doku koordinatları, parça düzenleme | Bölüm 6 (doku) |
| `threesweep/model.py` | Fotoğraf + parçalar, proje dosyası | – |
| `threesweep/export.py` | GLB / OBJ dışa aktarma | – |
| `threesweep/app.py` | Çizim arayüzü (PyQt6) | – |
| `threesweep/view3d.py` | Uygulama içindeki 3D görünüm ve parça düzenleme (pyqtgraph) | – |
| `threesweep/viewer.py` | Alternatif 3D görüntüleyici (Polyscope) | – |
| `threesweep/synthetic.py` | Gerçek boyutu bilinen yapay test fotoğrafları | – |
| `tests/` | Otomatik testler (boyut doğruluğu, mesh, dışa aktarma, arayüz) | – |

## Makaleden farklar ve basitleştirmeler

Bu sürüm uçtan uca çalışan bir başlangıç. Makaleye göre bilerek basitleştirilen yerler:

- **Kamera ortografik.** Makale perspektif kamera kullanıyor (varsayılan 45° görüş açısı).
  Ortografikte derinlik görüntüyü değiştirmediği için matematik çok sadeleşiyor. Perspektif
  eklenecekse `geometry.project` ve `sweep.py`'deki derinlik adımı değişmeli.
- **Kenar bulma Canny ile.** Makale gPb konturlarını kullanıyor. Daha iyi bir kenar
  haritası `EdgeMap.__init__` değiştirilerek eklenebilir.
- **Eğri eksenlerde eğim sabit.** Eksenin görüntü düzleminden çıkış açısı parça boyunca
  sabit tutuluyor. Düz eksenlerde bu kesin doğru. Makale de eğri eksenlerin büyük ölçüde
  görüntü düzlemine paralel olduğunu varsayıyor.
- **Parçalar arası kısıtlar yok.** Paralellik, eş doğrusallık, eş düzlemlilik optimizasyonu
  (Bölüm 5) henüz yazılmadı. Parçalar şimdilik birbirinden bağımsız konumlanıyor.
- **Post-snapping yok.** Makalede her parçadan sonra küçük bir hizalama araması yapılıyor (±%10).
- **Kopyala-yapıştır, küre primitifi, arka plan doldurma (inpainting) yok.**
- **Arka yüz dokusu:** ortografik izdüşüm sayesinde arka yüz ön yüzün aynası oluyor, makaledeki
  simetri varsayımıyla aynı sonuç. Görünmeyen yan yüzlerdeki boşluklar doldurulmuyor.

## Bilinen sorun: Windows Smart App Control ve Polyscope

Smart App Control açık olan Windows bilgisayarlarda Polyscope'un DLL'i engelleniyor
("An Application Control policy has blocked this file"). Bu yüzden uygulamanın 3D
görünümü pyqtgraph ile yazıldı ve Polyscope'a ihtiyaç duymuyor. Polyscope çalışan
bilgisayarlarda **View > Show 3D in Polyscope** ile o görüntüleyici de açılabilir.

## Sonraki adımlar için öneriler

1. Gerçek fotoğraflarla deneme (sade arka plan, net kenarlar), çıkan sorunları not etme
2. Perspektif kamera ve bakış açısı ayarı
3. Parçalar arası kısıt optimizasyonu (Bölüm 5): masa bacaklarının paralel olması gibi
4. Yeni özellikler: SAM ile nesne seçimi, monodepth ile derinlik önerisi, oyuna hazır çıktı
   (parçaları birleştirme, LOD, collider)
