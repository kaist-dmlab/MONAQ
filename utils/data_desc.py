data_contexts = {
    "BinaryHeartbeat": """The task is to classify the nature of the heartbeat signal.
Heart sound recordings were sourced from several contributors around the world, collected at either a clinical or nonclinical environment, from both healthy subjects and pathological patients. Each series represent the amplitude of the signal over time. The heart sound recordings were collected from different locations on the body. The typical four locations are aortic area, pulmonic area, tricuspid area and mitral area, but could be one of nine different locations.

The sounds were divided into two classes: normal and abnormal. 
The normal recordings were from healthy subjects and the abnormal ones were from patients with a confirmed cardiac diagnosis. The patients suffer from a variety of illnesses, but typically they are heart valve defects and coronary artery disease patients. Heart valve defects include mitral valve prolapse, mitral regurgitation, aortic stenosis and valvular surgery. All the recordings from the patients were generally labeled as abnormal. Both healthy subjects and pathological patients include both children and adults.

Data was recorded at 2,000Hz and truncated to the shortest instance to make it equal length.
Instances: 409
Time series length: 18,530
Classes:
- Normal (110)
- Abnormal (299)
Default train test split created through a random partition.""",
    "AppliancesEnergy": """Appliances Energy Dataset
The goal of this dataset is to predict total energy usage in kWh of a house. This dataset contains 138 time series obtained from the Appliances Energy Prediction dataset from the UCI repository. 
The time series has 24 dimensions. This includes temperature and humidity measurements of 9 rooms in a house, monitored with a ZigBee wireless sensor network. It also includes weather and climate data such as temperature, pressure, humidity, wind speed, visibility and dewpoint measured from Chievres airport. The data set is averaged for 10 minutes period and spanning 4.5 months.""",
    "AtrialFibrillation": """This is a physionet dataset of two-channel ECG recordings has been created from data used in the Computers in Cardiology Challenge 2004, an open competition with the goal of developing automated methods for predicting spontaneous termination of atrial fibrillation (AF).

The raw instances were 5 second segments of atrial fibrillation, containing two ECG signals, each sampled at 128 samples per second. The class labels are: n, s and t.

class n is described as a non termination artiral fibrilation(that is, it did not terminate for at least one hour after the original recording of the data).
class s is described as an atrial fibrilation that self terminates at least one minute after the recording process.
class t is described as terminating immediately, that is within one second of the recording ending.""",
    "LiveFuelMoistureContent": """The goal of this LiveFuelMoistureContent dataset is to predict the moisture content in vegetation. 
The moisture is the ratio between the weight of water in vegetation and the weight of the dry part of vegetation.  This is known as the live fuel moisture content (LFMC) and is an important variable as the risk of fire increases very rapidly as soon as the LFMC goes below 80%.This data is paired with one year of daily reflectance data at 7 spectral bands (459 nm to 2155 nm) before the LFMC sampling date from the Moderate Resolution Imaging Spectrometer (MODIS) satellite. 
This dataset contains 5003 time series obtained from researchers at Monash University with 7 dimensions, each corresponding to one spectral band.
""",
    "BenzeneConcentration": """This goal of this BenzeneConcentration dataset is to predict benzene concentration in an Italian city. 
This dataset contains 8878 time series obtained from the Air Quality dataset from the UCI repository. 
The time series has 8 dimensions which consists of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device, as well as temperature, relative humidity and absolute humidity. The Air Quality Chemical Multisensor device was located on the field in a significantly polluted area, at road level, within an Italian city. Data were recorded from March 2004 to February 2005 (one year) representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer.
""",
    "BIDMC32SpO2": """BIDMC Blood Oxygen Saturation Dataset (32 seconds window)
The goal of this dataset is to estimate blood oxygen saturation level using PPG and ECG data. 
This dataset contains 7949 time series obtained from the Physionet's BIDMC PPG and Respiration dataset, which was extracted from the much larger MIMIC II waveform database.""",
    "Cricket": """Cricket requires an umpire to signal different events in the game to a distant scorer/bookkeeper. 
The signals are communicated with motions of the hands. For example, No-Ball is signaled by touching each shoulder with the opposite hand, and TV-Replay, a request for an off-field review of the video of a play, is signaled by miming the outline of a TV screen. 

The dataset introduced consists of four umpires performing twelve signals, each with ten repetitions. 
The data, recorded at a frequency of 184Hz, was collected by placing accelerometers on the wrists of the umpires. Each accelerometer has three synchronous measures for three axes (X, Y and Z). 
Thus, we have a six-dimensional problem from the two accelerometers. 60% of the data is used for training and use the rest as testing data. The series are 1197 long.""",
    "FaultDetectionA": """FaultDetectionA is a subset taken from the FaultDetection, which is gathered from an electromechanical drive system that monitors the condition of rolling bearings and detect damages in them.
There are four subsets of data collected under various conditions, whose parameters include rotational speed, load torque, and radial force.

FaultDetectionA has three classes, the distribution of which is unbalanced, but identical in the train and test sets.
- undamaged (9.09%)
- inner damaged (45.55%)
- outer damaged (45.55%)

Each original recording has a single channel with sampling frequency of 64k Hz and lasts 4 seconds. The data we host was processed with a sliding window to produce more cases.
The data were split into 8184 train cases, 2728 validation and 2728 test. We have added the validation set to the end of the train file to ease reproduction if a validation set is needed. 
We are not aware of whether the original recordings were separated in the train and test.""",
    "FloodModeling": """Flood Modeling Dataset 1
The goal of this dataset is to predict maximum water depth for flood modelling.
The dataset contains 673 hourly rainfall events time series which are used to predict the maximum water depth of a domain (Digital Elevation Model, DEM). 
The rainfall events and DEM are generated synthetically by researchers at Monash University because real DEM data with accurate rainfall events are rare.""",
    "UCIHAR": """A human activity recognition data from the UCR archive. HAR contains recordings of 30 health volunteers aged 19-48 years old. The six classes are balanced and are
- walking
- walking upstairs
- walking downstairs
- sitting
- standing
- laying down.

The wearable sensors on a smartphone measure triaxial linear acceleration and triaxial angular velocity at 50 Hz. The UCI data has six channels. 
This data was preprocessed. It has just three channels representing the body linear acceleration.

The original UCI data has 10299 instances split into 70% train and 30% test. with separate subjects in train and test.
This data was split into train (5881 cases), validation (1471) and test (2947). We have added the validation set to the end of the train file to ease reproduction if a validation set is needed.""",
}

feature_descriptions = {
    "BinaryHeartbeat": """Classes:
- Normal
- Abnormal

Dimensions:
0. Heart sound recordings 
""",
    "AppliancesEnergy": """
Label: 
Appliances: Total energy use in kWh

Dimensions:
0. T1: Temperature in kitchen area, in Celsius (daily at 10 minutes interval)
1. RH_1: Humidity in kitchen area, in % (daily at 10 minutes interval)
2. T2: Temperature in living room area, in Celsius (daily at 10 minutes interval)
3. RH_2: Humidity in living room area, in % (daily at 10 minutes interval)
4. T3: Temperature in laundry room area (daily at 10 minutes interval)
5. RH_3: Humidity in laundry room area, in % (daily at 10 minutes interval)
6. T4: Temperature in office room, in Celsius (daily at 10 minutes interval)
7. RH_4: Humidity in office room, in % (daily at 10 minutes interval)
8. T5: Temperature in bathroom, in Celsius (daily at 10 minutes interval)
9. RH_5: Humidity in bathroom, in % (daily at 10 minutes interval)
10. T6: Temperature outside the building (north side), in Celsius (daily at 10 minutes interval)
11. RH_6: Humidity outside the building (north side), in % (daily at 10 minutes interval)
12. T7: Temperature in ironing room , in Celsius (daily at 10 minutes interval)
13. RH_7: Humidity in ironing room, in % (daily at 10 minutes interval)
14. T8: Temperature in teenager room 2, in Celsius (daily at 10 minutes interval)
15. RH_8: Humidity in teenager room 2, in % (daily at 10 minutes interval)
16. T9: Temperature in parents room, in Celsius (daily at 10 minutes interval)
17. RH_9: Humidity in parents room, in % (daily at 10 minutes interval)
18. T_out: Temperature outside (from Chievres weather station), in Celsius (daily at 10 minutes interval)
19. Pressure (from Chievres weather station): in mm Hg (daily at 10 minutes interval)
20. RH_out: Humidity outside (from Chievres weather station), in % (daily at 10 minutes interval)
21. Wind speed (from Chievres weather station): in m/s (daily at 10 minutes interval)
22. Visibility (from Chievres weather station): in km (daily at 10 minutes interval)
23. T_dewpoint (from Chievres weather station): °C (daily at 10 minutes interval)""",
    "AtrialFibrillation": """Classes:
- n is described as a non termination artiral fibrilation(that is, it did not terminate for at least one hour after the original recording of the data).
- s is described as an atrial fibrilation that self terminates at least one minuet after the recording process.
- t is descirbed as terminating immediatly, that is within one second of the recording ending.

Dimensions:
0. 1-D ECG signal
1. 1-D ECG signal""",
    "LiveFuelMoistureContent": """Label: 
LFMCvalue: Live fuel moisture content value

Dimensions:
0. Band 1: MODIS satellite band 1, sampled daily for a year
1. Band 2: MODIS satellite band 2, sampled daily for a year
2. Band 3: MODIS satellite band 3, sampled daily for a year
3. Band 4: MODIS satellite band 4, sampled daily for a year
4. Band 5: MODIS satellite band 5, sampled daily for a year
5. Band 6: MODIS satellite band 6, sampled daily for a year
6. Band 7: MODIS satellite band 7, sampled daily for a year""",
    "BenzeneConcentration": """Label: 
C6H6(GT): True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)

Dimensions:
0. PT08.S1(CO): PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted), 10 days at one hour interval 
1. PT08.S2(NMHC): PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted), 10 days at one hour interval 
2. PT08.S3(NOx): PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted), 10 days at one hour interval 
3. PT08.S4(NO2): PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted), 10 days at one hour interval 
4. PT08.S5(O3): PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted), 10 days at one hour interval 
5. T: Temperature in °C, 10 days at one hour interval 
6. RH: Relative Humidity (%), 10 days at one hour interval 
7. AH: Absolute Humidity, 10 days at one hour interval""",
    "BIDMC32SpO2": """Label:
SpO2: Blood oxygen saturation level sampled at 1 Hz.

Dimensions:
0. PPG: 32 seconds PPG sensor sampled at 125 Hz
1. ECG: 32 seconds ECG sensor sampled at 125 Hz""",
    "Cricket": """Classes:
- 12 motions of the hand signals

Dimensions:
0. left-wrist accelerometer X-axis
1. left-wrist accelerometer Y-axis 
2. left-wrist accelerometer Z-axis 
3. right-wrist accelerometer X-axis 
4. right-wrist accelerometer Y-axis 
5. right-wrist accelerometer Z-axis""",
    "FaultDetectionA": """Classes:
- undamaged
- inner damaged
- outer damaged

Dimensions:
0. 1-D signal of electromechanical drive""",
    "FloodModeling": """Label: 
Hmax: Maximum water depth of a Digital Elevation Model (DEM) calculated from Lisflood-FP solver

Dimensions:
0. Simulated rainfall for 266 hours""",
    "UCIHAR": """Classes:
- walking
- walking upstairs
- walking downstairs
- sitting
- standing
- laying down.

Dimensions:
0. body accelerometer X-axis
1. body accelerometer Y-axis 
2. body accelerometer Z-axis 
""",
}


AVAILABEL_DATASETS = list(data_contexts.keys())
CLS_DATASETS = [
    "BinaryHeartbeat",
    "AtrialFibrillation",
    "Cricket",
    "FaultDetectionA",
    "UCIHAR",
]
REG_DATASETS = [
    "AppliancesEnergy",
    "LiveFuelMoistureContent",
    "BenzeneConcentration",
    "BIDMC32SpO2",
    "FloodModeling",
]
TS_LENGTHS = {
    "BinaryHeartbeat": 18530,
    "AppliancesEnergy": 144,
    "AtrialFibrillation": 640,
    "LiveFuelMoistureContent": 365,
    "BenzeneConcentration": 240,
    "BIDMC32SpO2": 4000,
    "Cricket": 1197,
    "FaultDetectionA": 5120,
    "FloodModeling": 266,
    "UCIHAR": 206,
}

feature_names = {
    "BinaryHeartbeat": ["Heart sound recordings "],
    "AppliancesEnergy": [
        "T1",
        "RH_1",
        "T2",
        "RH_2",
        "T3",
        "RH_3",
        "T4",
        "RH_4",
        "T5",
        "RH_5",
        "T6",
        "RH_6",
        "T7",
        "RH_7",
        "T8",
        "RH_8",
        "T9",
        "RH_9",
        "T_out",
        "Pressure",
        "RH_out",
        "Wind speed",
        "Visibility",
        "T_dewpoint",
    ],
    "AtrialFibrillation": ["ECG signal", "ECG signal"],
    "LiveFuelMoistureContent": [
        "Band 1",
        "Band 2",
        "Band 3",
        "Band 4",
        "Band 5",
        "Band 6",
        "Band 7",
    ],
    "BenzeneConcentration": ["CO", "NMHC", "NOx", "NO2", "O3", "T", "RH", "AH"],
    "BIDMC32SpO2": ["PPG", "ECG"],
    "Cricket": [
        "left-wrist accelerometer X-axis",
        "left-wrist accelerometer Y-axis",
        "left-wrist accelerometer Z-axis",
        "right-wrist accelerometer X-axis",
        "right-wrist accelerometer Y-axis",
        "right-wrist accelerometer Z-axis",
    ],
    "FaultDetectionA": ["signal of electromechanical drive"],
    "FloodModeling": ["rainfall"],
    "UCIHAR": [
        "body accelerometer X-axis",
        "body accelerometer Y-axis",
        "body accelerometer Z-axis",
    ],
}
