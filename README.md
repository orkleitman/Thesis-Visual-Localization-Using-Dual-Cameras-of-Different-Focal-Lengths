# Visual Localization with Dual-Focal Cameras

[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Last Commit](https://img.shields.io/github/last-commit/orkleitman/Thesis-Visual-Localization-Using-Dual-Cameras-of-Different-Focal-Lengths)

**Problem:** Accurate 3-D motion from vision-only is challenging indoors.  
**Approach:** Dual cameras (Wide + Ultra Wide), SIFT + RANSAC, 3-D triangulation, gradient-descent multi-start optimization.  
**Results:** Sub-pixel reprojection error on controlled sequences; robust to side and rotational motion.  
**Why it matters:** Transparent, sensor-free pipeline for robotics, AR, and indoor navigation.

➡ לפרטים מלאים בעברית — גללו מטה.

---

## Quickstart

```bash
pip install -r requirements.txt
python code/motion_estimation/motion_estimation.py --input frames/Back_0-5-10-15_cm --results_dir results




# עבודת תזה – לוקליזציה חזותית באמצעות זוג מצלמות עם מוקדים שונים

ריפו זה כולל את הקוד, קבצי הקלט (frames), ותוצאות הניסויים (results) עבור עבודת התזה שלי.  
המערכת מבצעת שיערוך תנועה (Motion Estimation) ללא חיישנים נוספים, על בסיס זוג מצלמות עם מוקדים שונים (Wide + Ultra Wide).  

---

## 📂 מבנה התיקיות

```plaintext
Thesis-Visual-Localization/
├── code/                               # קוד מקור
│   └── motion_estimation/
│       └── motion_estimation.py        # הקובץ הראשי עם האלגוריתם
│
├── frames/                             # קבצי הקלט (תמונות לפי סוג תנועה)
│   ├── Back_0-5-10-15_cm/
│   ├── Forward_0-5-10-15_cm/
│   ├── Left_0-5-10-15_cm/
│   ├── Right_0-5-10-15_cm/
│   ├── Upward_0-5-10-15_cm/
│   ├── Downward_0-5-10-15_cm/
│   ├── Roll_Clockwise_0-10-20-30_degrees/
│   ├── Pitch_Clockwise_0-10-20-30_degrees/
│   ├── Counterclockwise_0-10-20-30_degrees/
│   └── Clockwise_0-10-20-30_degrees/
│       └── (בכל תיקייה יש 8 פריימים – 4 מהמצלמה Wide ו־4 מהמצלמה Ultra)
│
├── results/                            # פלטים מההרצות
│   ├── <motion>_<date>_<time>/         # תיקייה לכל ניסוי (לפי תנועה + חותמת זמן)
│   │   ├── sift_features/              # תמונות עם נקודות SIFT מסומנות
│   │   ├── visualizations/             # גרפים של התאמות והתכנסות
│   │   └── analysis_report.txt         # דוח טקסט לכל ניסוי
│   └── master_summary_*.txt            # קובץ סיכום כולל לכל הניסויים
│
├── .gitignore                          # קובץ להתעלמות מקבצים מיותרים ב־Git
├── requirements.txt                    # ספריות פייתון נדרשות להרצה
└── README.md                           # הקובץ המתאר את הפרויקט

---

## 📸 קלט – Frames

- כל ניסוי מיוצג בתיקייה נפרדת תחת `frames/`.  
- בכל תיקייה יש **8 פריימים**:  
  - 4 תמונות מהמצלמה **Wide**  
  - 4 תמונות מהמצלמה **Ultra Wide**  
- שמות הקבצים כוללים:  
  - מילה `wide` או `ultra`  
  - מספר פריים `f1`–`f4`  
- דוגמה:
```
wide_middle_vertical_f1.jpg  
ultra_middle_vertical_f1.jpg
```

---

## ⚙️ התקנה

יש להתקין את הספריות המופיעות ב־`requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 הרצה

### אפשרות א' – שימוש בקוד ישירות (בתוך Python)
```python
from code.motion_estimation.motion_estimation import TransparentMotionAnalyzer

analyzer = TransparentMotionAnalyzer(results_base_dir="results")
motion = analyzer.analyze_sequence("frames/Back_0-5-10-15_cm")
print("Detected motion:", motion)
```

### אפשרות ב' – הרצת הקובץ הראשי (מתוך טרמינל)
```bash
python code/motion_estimation/motion_estimation.py
```

*(במידת הצורך ערכי קודם בקובץ `motion_estimation.py` את רשימת הנתיבים `test_paths` שיצביעו על התיקיות הרצויות ב־`frames/`).*

---

## 📊 פלטים (Results)

- לכל ניסוי נוצרת תיקייה חדשה תחת `results/` עם שם התנועה וחותמת זמן.  
- בכל תיקייה:
  - `sift_features/` – תמונות עם נקודות SIFT מסומנות  
  - `visualizations/` – גרפים של התאמות ועקומות התכנסות  
  - `analysis_report.txt` – דוח טקסט מפורט לכל ניסוי  
- בנוסף, יש קובץ סיכום כללי:  
  - `master_summary_*.txt` – מרכז את כלל תוצאות הניסויים  

---

## 🧠 שיטת העבודה (בקצרה)

1. זיהוי נקודות תכונה באמצעות **SIFT**  
2. סינון התאמות בעזרת **RANSAC**  
3. טריאנגולציה Wide–Ultra ליצירת נקודות תלת־ממדיות  
4. אופטימיזציה באמצעות **Gradient Descent** להפחתת שגיאת הקרנה  
5. חיפוש גלובלי (Multi-start) עם שיפור הדרגתי של ה־learning rate  
6. הפקת גרפים, ויזואליזציות ודוחות טקסט לכל ניסוי  

---

## 📜 הערות

- ניתן לשנות את תיקיית הפלט (`results`) ע"י פרמטר `results_base_dir`.  
- קובץ `.gitignore` מוודא שקבצים מיותרים (cache, npy, npz, csv) לא נשמרים בגיט.  
- קיימים קובצי LICENSE ו־CITATION.cff כך שאפשר להשתמש בקוד ולצטט בצורה נכונה.
- 
---
