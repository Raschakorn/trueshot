# TrueShot V3 - World Class AI vs Real Detector

## ทำไมถึงเรียกว่าระดับโลก

ของเดิม (V1) ที่ทุกคนทำกันคือ เอาโมเดลตัวเดียวมาทาย
ของ V3 คือ **ระบบ 5 ชั้น** แบบที่งานวิจัยระดับโลกใช้กัน:

1. **Provenance** - C2PA + SynthID + EXIF
   - C2PA = ลายเซ็นดิจิทัลจาก Adobe/Google/Meta ปลอมไม่ได้
   - SynthID = ลายน้ำที่ Google ฝังในภาพ AI
   - ถ้ามี 2 อันนี้ คือจบ ไม่ต้องเดา

2. **Pixel Forensic** - ELA + FFT + PRNU + Noise
   - ดูระดับบิท ไม่ใช่ดูด้วยตา

3. **Semantic** - Thai Text + Hand/Face + Physics
   - จุดอ่อนของ AI โลกคือภาษาไทย + มือ + แสงเงา
   - นี่คือข้อได้เปรียบของไทยที่จะชนะฝรั่ง

4. **Ensemble AI** - 3 โมเดลโหวตกัน
   - ไม่เชื่อโมเดลตัวเดียว ใช้ umm-maybe + Organika/sdxl-detector + DINOv2 โหวต
   - ถ้าโมเดลไม่เห็นตรงกัน = ตีว่าก้ำกึ่ง ไม่ฟันธงมั่ว

5. **Explainable** - บอกว่าเพราะอะไร
   - งานระดับโลกต้องอธิบายได้ ไม่ใช่บอกแค่ตัวเลข
   - มี Heatmap ELA / Noise ให้คนตรวจตามได้

## วิธีรัน
```bash
pip install streamlit pillow numpy opencv-python-headless
streamlit run app_v3.py
```

## Roadmap ไปแข่งระดับโลกจริง
- [ ] เพิ่มชุดข้อมูลเทส GenImage + CIFAKE (120k รูป)
- [ ] Benchmark เทียบกับ Hive / Illuminarty
- [ ] เพิ่ม OCR ภาษาไทย (EasyOCR) ตรวจตัวอักษรไทยพัง
- [ ] เปิด Open Source + ให้คนทั่วโลกช่วยส่งรูปมาเทส
- [ ] ทำ API + Browser Extension + Mobile App

## ปรัชญา
> อย่าไล่จับ AI แต่สร้างระบบที่พิสูจน์ว่าจริงได้
> ภาพจริงต้องมีหลักฐาน ภาพ AI ต้องไม่มีหลักฐาน
