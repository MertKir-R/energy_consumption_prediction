# 1. Use an official, lightweight Python image as the base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
# (We do this first because Docker caches this step, making future builds way faster!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your specific model artifacts into the exact same folder structure
COPY runs/xgb_run_01/model.joblib runs/xgb_run_01/
COPY runs/xgb_run_01/train_columns.json runs/xgb_run_01/

# 5. Copy your API code
COPY src/app.py src/

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Command to run the API when the container starts
# (Notice we added --host 0.0.0.0 so it can talk to the outside world)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]