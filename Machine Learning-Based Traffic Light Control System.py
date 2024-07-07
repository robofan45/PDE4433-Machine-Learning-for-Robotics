import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
NUM_LANES = 4
MAX_VEHICLES = 20
SIMULATION_STEPS = 2000

def load_data():
    try:
        traffic_two_month = pd.read_csv(r"C:\Users\MAHIM TRIVEDI\OneDrive\Documents\Downloads\TrafficTwoMonth.csv")
        traffic = pd.read_csv(r"C:\Users\MAHIM TRIVEDI\OneDrive\Documents\Downloads\Traffic.csv")
        
        # Combine traffic data
        combined_traffic = pd.concat([traffic_two_month, traffic], axis=0, ignore_index=True)
        
        # Convert 'Time' to datetime if it's not already
        combined_traffic['Time'] = pd.to_datetime(combined_traffic['Time'])
        
        # Extract hour from Time
        combined_traffic['Hour'] = combined_traffic['Time'].dt.hour
        
        # One-hot encode 'Day of the week'
        day_dummies = pd.get_dummies(combined_traffic['Day of the week'], prefix='Day')
        combined_traffic = pd.concat([combined_traffic, day_dummies], axis=1)
        
        # Select features and target
        features = ['Hour', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount'] + list(day_dummies.columns)
        target = 'Total'
        
        X = combined_traffic[features].values
        y = combined_traffic[target].values
        
        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y, scaler
    
    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        raise

def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Single output for predicting total traffic volume
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

class Vehicle:
    def __init__(self, lane):
        self.lane = lane
        self.position = 0
        self.speed = np.random.randint(1, 5)
        self.size = np.random.randint(1, 3)
        self.waiting_time = 0
        
    def move(self, traffic_light_state):
        if traffic_light_state[self.lane] == 1:  # Green light
            self.position += self.speed
            self.waiting_time = 0
        else:  # Red light
            if self.position < 95:
                self.position = min(self.position + self.speed, 95)
                self.waiting_time += 1

class Intersection:
    def __init__(self):
        self.lanes = [[] for _ in range(NUM_LANES)]
        self.traffic_light_state = [0, 0, 0, 0]
        self.total_waiting_time = 0
        self.vehicles_passed = 0
        
    def add_vehicle(self, lane):
        if len(self.lanes[lane]) < MAX_VEHICLES:
            self.lanes[lane].append(Vehicle(lane))
    
    def remove_vehicles(self):
        for lane in range(NUM_LANES):
            passed_vehicles = [v for v in self.lanes[lane] if v.position > 100]
            self.vehicles_passed += len(passed_vehicles)
            self.lanes[lane] = [v for v in self.lanes[lane] if v.position <= 100]
    
    def update(self, predicted_volume):
        # Simple logic: set green light for the lane with highest predicted volume
        self.traffic_light_state = [0, 0, 0, 0]
        self.traffic_light_state[np.argmax(predicted_volume)] = 1
        
        for lane in range(NUM_LANES):
            for vehicle in self.lanes[lane]:
                vehicle.move(self.traffic_light_state)
                self.total_waiting_time += vehicle.waiting_time
        
        self.remove_vehicles()

class TrafficLight:
    def __init__(self, ax, position, direction):
        self.position = position
        self.direction = direction
        self.lights = []
        
        if direction in ['N', 'S']:
            x_offset = 0
            y_offset = -5 if direction == 'N' else 5
        else:
            x_offset = -5 if direction == 'E' else 5
            y_offset = 0
        
        colors = ['red', 'yellow', 'green']
        for i, color in enumerate(colors):
            light = Circle((position[0] + x_offset, position[1] + y_offset + i*2.5), 1, facecolor=color, edgecolor='black')
            ax.add_patch(light)
            self.lights.append(light)
    
    def set_state(self, state):
        for light in self.lights:
            light.set_visible(False)
        if state == 0:  # Red
            self.lights[0].set_visible(True)
        elif state == 1:  # Green
            self.lights[2].set_visible(True)

def run_simulation(model, X_test, scaler):
    intersection = Intersection()
    
    fig, (ax_intersection, ax_stats) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Setup intersection visualization
    ax_intersection.set_xlim(-10, 110)
    ax_intersection.set_ylim(-10, 110)
    ax_intersection.set_aspect('equal', adjustable='box')
    ax_intersection.axis('off')

    # Create road
    ax_intersection.add_patch(Rectangle((-10, 40), 120, 20, facecolor='gray'))
    ax_intersection.add_patch(Rectangle((40, -10), 20, 120, facecolor='gray'))

    # Create traffic lights
    traffic_lights = [
        TrafficLight(ax_intersection, (45, 60), 'N'),
        TrafficLight(ax_intersection, (55, 40), 'S'),
        TrafficLight(ax_intersection, (40, 55), 'W'),
        TrafficLight(ax_intersection, (60, 45), 'E')
    ]

    # Setup statistics visualization
    ax_stats.set_xlim(0, SIMULATION_STEPS)
    ax_stats.set_ylim(0, MAX_VEHICLES * NUM_LANES)
    ax_stats.set_xlabel('Simulation Step')
    ax_stats.set_ylabel('Number of Vehicles')
    ax_stats.set_title('Traffic Statistics')
    ax_stats.grid(True)

    waiting_line, = ax_stats.plot([], [], label='Waiting Vehicles')
    passed_line, = ax_stats.plot([], [], label='Passed Vehicles')
    ax_stats.legend()

    # Vehicle representations
    vehicles = [ax_intersection.add_patch(Rectangle((0, 0), 4, 2, facecolor='red', visible=False))
                for _ in range(MAX_VEHICLES * NUM_LANES)]

    waiting_vehicles = []
    passed_vehicles = []

    def init():
        return vehicles + [waiting_line, passed_line]

    def update(frame):
        # Use real data for simulation state
        state = X_test[frame % len(X_test)]
        
        # Predict traffic volume for each lane
        predicted_volumes = []
        for lane in range(NUM_LANES):
            lane_state = state.copy()
            lane_state[0] = lane  # Set lane number
            predicted_volume = model.predict(lane_state.reshape(1, -1))[0][0]
            predicted_volumes.append(predicted_volume)
        
        # Add vehicles based on predicted volume
        for lane, volume in enumerate(predicted_volumes):
            if np.random.random() < volume / 1000:  # Adjust this threshold as needed
                intersection.add_vehicle(lane)
        
        intersection.update(predicted_volumes)
        
        # Update traffic lights
        for i, light in enumerate(traffic_lights):
            light.set_state(intersection.traffic_light_state[i])
        
        # Update vehicle positions
        for v in vehicles:
            v.set_visible(False)
        
        v_index = 0
        for lane in range(NUM_LANES):
            for vehicle in intersection.lanes[lane]:
                if v_index < len(vehicles):
                    if lane == 0:  # Bottom to top
                        vehicles[v_index].set_xy((45, vehicle.position))
                    elif lane == 1:  # Top to bottom
                        vehicles[v_index].set_xy((55, 100 - vehicle.position))
                    elif lane == 2:  # Left to right
                        vehicles[v_index].set_xy((vehicle.position, 55))
                    else:  # Right to left
                        vehicles[v_index].set_xy((100 - vehicle.position, 45))
                    vehicles[v_index].set_facecolor('red' if vehicle.size == 1 else 'blue')
                    vehicles[v_index].set_visible(True)
                    v_index += 1

        # Update statistics
        waiting_vehicles.append(sum(len(lane) for lane in intersection.lanes))
        passed_vehicles.append(intersection.vehicles_passed)
        waiting_line.set_data(range(frame + 1), waiting_vehicles)
        passed_line.set_data(range(frame + 1), passed_vehicles)

        return vehicles + [waiting_line, passed_line]

    anim = FuncAnimation(fig, update, frames=SIMULATION_STEPS, init_func=init, interval=50, blit=True)
    plt.tight_layout()
    plt.show()

    print(f"Total vehicles passed: {intersection.vehicles_passed}")
    print(f"Average waiting time: {intersection.total_waiting_time / intersection.vehicles_passed:.2f}")

# Main execution
if __name__ == "__main__":
    try:
        # Load and preprocess data
        X, y, scaler = load_data()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = create_model(input_shape=(X_train.shape[1],))
        model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

        # Evaluate the model
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test MAE: {test_mae:.3f}")

        # Run the simulation with real test data
        run_simulation(model, X_test, scaler)

    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
