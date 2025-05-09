import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import json
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, jsonify, request

# Configuration
SAMPLE_SIZE = 100
CALIBRATION_READINGS = 20

class SmartSpoon:
    def __init__(self):
        # Simulated sensor calibration values
        self.calibration = {
            "no_salt": 0.1,  # baseline reading
            "low_salt": 0.3,  # low salt concentration
            "medium_salt": 0.6,  # medium salt concentration
            "high_salt": 0.9   # high salt concentration
        }
        
        # Initialize data storage
        self.readings = []
        self.is_calibrated = False
        self.model = None
        
    def calibrate(self):
        """Calibrate the smart spoon with known salt concentrations"""
        print("Starting calibration sequence...")
        
        calibration_data = []
        
        # Collect readings for each known salt concentration
        for salt_level, true_value in self.calibration.items():
            print(f"Calibrating for {salt_level}...")
            
            # Simulate collecting multiple readings at this salt level
            for _ in range(CALIBRATION_READINGS):
                # Add some noise to simulate real readings
                reading = true_value + random.normalvariate(0, 0.05)
                calibration_data.append((reading, true_value))
                time.sleep(0.01)  # Short delay to simulate reading time
                
        # Train a simple linear regression model for calibration
        X = np.array([x[0] for x in calibration_data]).reshape(-1, 1)
        y = np.array([x[1] for x in calibration_data])
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        self.is_calibrated = True
        print("Calibration complete!")
        
        # Return calibration metrics
        score = self.model.score(X, y)
        return {
            "calibration_score": score,
            "coefficient": float(self.model.coef_[0]),
            "intercept": float(self.model.intercept_)
        }
    
    def read_salt_level(self):
        """Simulate reading salt level from the spoon sensor"""
        if not self.is_calibrated:
            raise Exception("Smart spoon needs to be calibrated first!")
        
        # Simulate a sensor reading (in real life this would come from hardware)
        # This simulates a random food sample with varying salt
        raw_reading = random.uniform(0.05, 0.95)
        
        # Apply calibration model to get true salt concentration
        calibrated_reading = self.model.predict([[raw_reading]])[0]
        
        # Ensure reading is within valid range
        calibrated_reading = max(0, min(1, calibrated_reading))
        
        # Store the reading
        timestamp = time.time()
        self.readings.append({
            "timestamp": timestamp,
            "raw_reading": raw_reading,
            "calibrated_reading": calibrated_reading,
            "salt_category": self._categorize_salt(calibrated_reading)
        })
        
        return self.readings[-1]
    
    def _categorize_salt(self, reading):
        """Categorize salt level based on reading"""
        if reading < 0.2:
            return "No salt"
        elif reading < 0.4:
            return "Low salt"
        elif reading < 0.7:
            return "Medium salt"
        else:
            return "High salt"
    
    def collect_sample_data(self, sample_size=SAMPLE_SIZE):
        """Collect a sample of readings"""
        print(f"Collecting {sample_size} samples...")
        
        for _ in range(sample_size):
            self.read_salt_level()
            time.sleep(0.01)  # Simulate reading delay
            
        print("Sampling complete!")
        return self.get_analytics()
    
    def get_analytics(self):
        """Analyze collected readings"""
        if not self.readings:
            return {"error": "No readings collected yet"}
        
        df = pd.DataFrame(self.readings)
        
        # Basic statistics
        stats = {
            "count": len(df),
            "mean_salt": float(df["calibrated_reading"].mean()),
            "median_salt": float(df["calibrated_reading"].median()),
            "std_dev": float(df["calibrated_reading"].std()),
            "min_salt": float(df["calibrated_reading"].min()),
            "max_salt": float(df["calibrated_reading"].max()),
        }
        
        # Salt category distribution
        category_counts = df["salt_category"].value_counts().to_dict()
        
        return {
            "stats": stats,
            "categories": category_counts,
            "readings": self.readings[-10:]  # Return only the most recent readings
        }
    
    def export_data(self, filename="smart_spoon_data.csv"):
        """Export collected data to CSV"""
        if not self.readings:
            return {"error": "No data to export"}
            
        df = pd.DataFrame(self.readings)
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
        return {"status": "success", "filename": filename}
    
    def generate_plots(self):
        """Generate plots for visualization"""
        if not self.readings:
            return {"error": "No data to plot"}
            
        df = pd.DataFrame(self.readings)
        
        # Create a timestamp for the plot filenames
        timestamp = int(time.time())
        
        # Plot 1: Salt readings over time
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(df)), df["calibrated_reading"], marker='o', linestyle='-', alpha=0.7)
        plt.title("Salt Concentration Readings")
        plt.xlabel("Reading Number")
        plt.ylabel("Salt Concentration (0-1)")
        plt.grid(True, alpha=0.3)
        plot_file1 = f"salt_readings_{timestamp}.png"
        plt.savefig(plot_file1)
        
        # Plot 2: Distribution of salt readings
        plt.figure(figsize=(10, 6))
        plt.hist(df["calibrated_reading"], bins=20, alpha=0.7, color='green')
        plt.title("Distribution of Salt Readings")
        plt.xlabel("Salt Concentration")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plot_file2 = f"salt_distribution_{timestamp}.png"
        plt.savefig(plot_file2)
        
        # Plot 3: Salt categories pie chart
        plt.figure(figsize=(10, 6))
        category_counts = df["salt_category"].value_counts()
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Salt Level Categories")
        plot_file3 = f"salt_categories_{timestamp}.png"
        plt.savefig(plot_file3)
        
        print(f"Plots generated: {plot_file1}, {plot_file2}, {plot_file3}")
        return {
            "time_series": plot_file1,
            "distribution": plot_file2,
            "categories": plot_file3
        }

# Flask web application for the smart spoon interface
app = Flask(__name__)
spoon = SmartSpoon()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calibrate', methods=['POST'])
def calibrate_spoon():
    results = spoon.calibrate()
    return jsonify(results)

@app.route('/read', methods=['GET'])
def read_salt():
    try:
        reading = spoon.read_salt_level()
        return jsonify(reading)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/collect', methods=['POST'])
def collect_data():
    sample_size = request.json.get('sample_size', SAMPLE_SIZE)
    results = spoon.collect_sample_data(sample_size)
    return jsonify(results)

@app.route('/analytics', methods=['GET'])
def get_analytics():
    results = spoon.get_analytics()
    return jsonify(results)

@app.route('/export', methods=['POST'])
def export_data():
    filename = request.json.get('filename', 'smart_spoon_data.csv')
    result = spoon.export_data(filename)
    return jsonify(result)

@app.route('/generate_plots', methods=['POST'])
def generate_plots():
    results = spoon.generate_plots()
    return jsonify(results)

# Save this code to a file named app.py
if __name__ == "__main__":
    print("Starting Smart Spoon Salt Detector application...")
    app.run(debug=True)