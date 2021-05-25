# Pendeteksi Manipulasi Citra
IF4073 Interpretasi dan Pengolahan Citra
Makalah UAS 2021

## Requirement
* [Python3] 
* [PIP]
* Python Libraries:
  * [exifread]
  * [opencv-python]
  * [progressbar2]
  * [Numpy]
  * [Scipy]
  * [Pillow]
  * [PyWavelets]
  * [Matplotlib]

Semua requirements dapat diinstall dengan menjalan perintah berikut di bash:
*./install_packet.sh*
 
## Fungsi-Fungsi
### 1. Analisis Metadata
```
python forensic.py cloning.jpg
```
atau:
```
python forensic.py -e cloning.jpg
```


### 2. JPEG Ghost
```
python forensic.py -g cloning.jpg
```

Untuk menggunakan serangkaian nilai kualitas:

```
python forensic.py -gm cloning.jpg
```

Daerah hasil manipulasi biasanya akan ditandai dengan border terang atau ditandai dengan daerah yang lebih gelap.

NB: Nilai kualitas penyimpanan gambar dapat diatur dengan perintah:
```
python forensic.py -g -q 50 demo.jpg
```

### 3. Noise Inconsistencies
```
python forensic.py -n cloning.jpg
```
Daerah hasil manipulasi biasanya akan ditandai dengan border terang atau ditandai dengan daerah yang lebih gelap.

Untuk menjalankan dengan fitur median-filter, jalankan:
```
python forensic.py -mn cloning.jpg
```

### 4. Error Level Analysis
```
python forensic.py -el cloning.jpg
```
Daerah hasil manipulasi biasanya akan ditandai dengan border terang atau ditandai dengan daerah yang lebih gelap.
