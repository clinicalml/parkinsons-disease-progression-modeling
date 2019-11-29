# PPMI data extraction

To extract data from PPMI:
1. Run `PPMI_Feature_Extraction_3_using_CMEDTM.py` passing PPMI directory in as first parameter and download date in format YYYYMmmDD as second parameter. Original csvs are expected to be in raw_data_asof_YYYYMmmDD directory within PPMI directory. Data will be outputted to pipeline_output directory within PPMI directory. The file contains all feature with one row per date per patient.<br>
`PPMI_Feature_Extraction.py` was tested on data downloaded on 2018Jun12. `PPMI_Feature_Extraction_3_using_CMEDTM.py` was tested on data downloaded on 2019Jan24 (contains some new data features)
2. Run `PPMI_Aggregate_Each_Visit_using_CMEDTM.py` with same parameter. Outputs 1 row per visit per patient to visit_feature_inputs directory.
3. Run `PPMI_Treatment_Extraction.py` with same parameter. Outputs 1 row per visit per patient with binary indicator for current treatment since last visit or since 3 months prior to first visit. Also outputs treatment history up to 3 months prior to first visit.

Code to reproduce the tables and figures in Ch. 2 is available in `Thesis data tables and figures.ipynb`