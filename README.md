# Near-miss-accident-prediction-using-dashcam-videos

**1) Please, download the "trace_model.pt" and "dht_r50_nkl_d97b97138.pth" (Deephough pretrain model) and put them into our project:**
Link: https://drive.google.com/file/d/1_e8QnFqQ7YeU8Vl140lSlvlADE_YD6oJ/view?usp=sharing

![image](https://github.com/keeganhuynh/Near-miss-accident-prediction-using-dashcam-videos/assets/58461941/1b732d76-bc6c-4dee-9924-792e6b12ef1a)

**2) Run the run.py twice if you get error: "no module name deep-hough"**

**3) HOW TO RUN THE VIDEO**

Input: video + kml_flie

Output: in you runs folder
- vanishing_point_list
- velocity_list
- risk_json
- data_json
- output_video

*In folder Runs, i give you a sample video and vanishing_point.txt to run the tool with out KML file, but you have to put "#" in front of the line 117 () (run.py)*
