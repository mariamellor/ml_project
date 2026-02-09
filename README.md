## ml_project

# RUNING FILES

The main analysis requires 3 scripts to be run in the following order: 
010-data_integration.py
020-model_building.py
030-predictions.py

We also included the code for the 3 models we built but chose not to use as our final model. 
021-model_building_RF_baseline.py
022-model_building_RF.py
023-model_buiding_adv_imput.py

# SPECIAL IMPUTER/ENCODER

We also created 9 separate files for specific encoder and imputers, which are listed below: 
__fn__AAV2020Encoder.py
__fn__ActivityTypeImputer.py
__fn__EmployeeImputer.py
__fn__FarmerImputer.py
__fn__RetiredEmployeeImputer.py
__fn__RetiredFarmerImputer.py
__fn__RetiredSelfEmployedImputer.py
__fn__SelfEmployedImputer.py
__fn__SportImputer.py

# DESCRIPTIVE STATISTICS

To create Figure 1 and Figure 2 you can run the following notebook: draft-stats_desc.ipynb

# ADDITIONAL DATA

We included additional data that can be downloaded from the following links. 
INSEE, Base 2025 - Aire d’Attraction des Villes: https://www.insee.fr/fr/statistiques/fichier/4803954/AAV2020_au_01-01-2025.zip
Natural Earth (map borders): https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
