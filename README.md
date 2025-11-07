# energy_consumption_prediction

**Forecasting yearly oil consumption for each country using XGBoost**

---

## Problem & Goal
- **Task:** Predict **yearly oil consumption** for **each EU counrtry**.
- **Why it matters:** Better forecasts help with planning of **energy** (specifically oil) need for EU countries .
- **Models used:** **XGBoost**

---

## Data
- **Source:** (https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption/data)
- **Cleaning:**
  - Only EU member countries included in the analysis.
  - records with NA oil_consumption eliminated from the dataset.
  - countries having oil_consumption data only after 1980 were eliminated (some former USSR countries like Lithuania, Latvia etc. eliminated)
  - for gdp external data from world bank is used, however if for that particular year-country world bank did not have data, gdp records in the actual dataset is used.
  - linear interpolation is used for missing values which are not trailing or leading (ie 123 NA NA 456 -> NA values replaced with linear interpolation) (function named linear_interpolation can be found in features.py)
  - linear extrapolation is used for missing values which are trailing (ie 123 456 NA -> NA values replaced with linear extrapolation). Last 7 years' values used for linear extrapolation (function named fill_extrapolation can be found in features.py)
- **Target:** `oil_consumption`
- **Features (examples):**
  - because of the high correlation consumption variables ('solar_consumption','coal_consumption','gas_consumption','biofuel_consumption','hydro_consumption','other_renewable_consumption','wind_consumption','nuclear_consumption') were considered for extended lag features (t-1, t-3, t-5). Again correlation with oil_consumption checked with these lag featured variables and 'gas_consumption', 'coal_consumption' found to be the ones that were most promising so extended lag features applied to these variables alongside with oil_consumption. For the rest of the variables t-1 years used assuming all variables are realized at the same time as oil_consumption (except for gdp and population) rolling means (last 3 years, 5 years, 7 years) were applied to 'gdp_wb','population','oil_consumption', 'gas_consumption', 'coal_consumption' variables. (related functions can be found in features.py functions named lag_feature and avg_lag_feature) 
- **Important:** No raw data is committed to Git.

---

## Structure
- After running an initial XGBoost model with random parameters to see get an idea of what range of parameters can be included, optimize.py script have been created to use Bayesian Optimization to find the parameters yielded the smallest validation rmse.
- Then train.py is created to see whether there is overfitting and how train/validation rmse behaves. Lastly model is saved here (model.joblib) with all the model trees, learned weights, hyperparameters etc.
- Lastly evaluate.py is created to see how model performs on the test/unseen data 

---

## Results
- Final model had 87 test RMSE, considering the oil_consumption in test set had mean 290 and std dev 330, we can say model did a good job. Also for some countries the predictions are plotted and can be seen that model can identify trends/correctly however the performance of the model should be evaluated by a domain expert.
- XGBoost assigned the highest gain importance to population, meaning that splits involving population contributed the most to reducing the model’s error. The next most influential features include oil_consumption_lag1 and long-term averages such as coal_consumption_avg_lag7, indicating that both recent consumption patterns and multi-year coal trends help the model make better predictions. Several country dummy variables (e.g., Sweden, Netherlands, Spain) also appear among the top features. This suggests that the model benefits from applying country specific adjustments that are not fully explained by the other variables.
