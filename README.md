# 2021-explainable-ai
This project provides supplementary code for academic paper.
The source code for barcode-like timeline is available at [https://github.com/rafcc/2020-prenatal-sono](https://github.com/rafcc/2020-prenatal-sono) 

##
Source data for all tables and figures are souce_data.xls

##
Procedure:
- process fetal heart screening videos using [SONO](https://www.mdpi.com/2076-3417/11/1/371) (tiny samples are provided in barcodeliketimeline_samples.zip)
- fill all pathes in ae_main.py    
- activate anaconda env using aetf1.yml  
- run ae_main.py    
- training and inference process begin and graph chart diagrams are obtained
- graph chart diagrams obtained by one trial are provided in chd_output_main_results.zip and normal_output_main_results.zip
- polygon_roc.py evaluates graph chart diagrams and outputs abnormality scores
