# Meal Intake Detection based on variations in Glucose Level

### Part 1

##### Input -
Five cell arrays:
1. The first cell array has tissue glucose levels every 5 mins for 2.5 hrs during a lunch meal. The data starts from 30 mins before meal intake an continues up to 2 hrs after the start of meal consumption. There are several such time series for one subject.
2. The second cell array has time stamps of each time series in the first cell array.
3. The third cell array has insulin basal infusion input time series at different times during the 2.5 hr time interval.
4. The fourth cell array has time stamps for each basal or bolus insulin delivery time series.
5. The fifth cell array has insulin bolus infusion input time series at different times during the 2.5 hr time interval.

##### Output -
1. Five different types of time series feature extractions from the CGM data cell array and the CGM timestamp cell array - Fast Fourier Transform, Discrete Wavelet Transform, Coefficient of Variation, Windowed Entropy, and Area Under Curve.
2. Feature matrix where each row is a collection of features from each time series.
3. Principal Component Analysis to derive the new feature matrix consisting of the top 5 features for each time series.

##### Steps -
1. Put .ipynb file in the same directory as patient data.
2. Open .ipynb file in Jupyter Notebook and execute.