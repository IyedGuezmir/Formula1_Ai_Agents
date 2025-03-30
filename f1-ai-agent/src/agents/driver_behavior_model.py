 # f1-ai-agent/src/agents/driver_behavior_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List

class DriverBehaviorModel:
    def __init__(self, driver_data: Dict):
        """
        Initialize Driver Behavior Model
        
        Args:
            driver_data (Dict): Historical driver performance data
        """
        self.driver_data = driver_data
        self.behavior_model = self._build_behavior_model()
    
    def _build_behavior_model(self):
        """
        Build neural network for driver behavior prediction
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax') 
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict_driver_actions(self, race_conditions: Dict) -> Dict:
        """
        Predict driver actions based on race conditions
        
        Args: 
            race_conditions (Dict): Current race conditions 
        
        Returns:
            Dict: Predicted driver actions and probabilities
        """
        input_features = self._prepare_input_features(race_conditions)
        action_probabilities = self.behavior_model.predict(input_features)[0]
        
        actions = [
            'aggressive_overtake',
            'defensive_driving',
            'consistent_pace',
            'risk_management',
            'strategic_positioning'
        ]
        
        return {
            'predicted_action': actions[np.argmax(action_probabilities)],
            'action_confidences': dict(zip(actions, action_probabilities))
        }
    
    def _prepare_input_features(self, race_conditions: Dict) -> np.ndarray:
        """
        Prepare input features for behavior prediction
        
        Args:
            race_conditions (Dict): Current race conditions
        
        Returns:
            np.ndarray: Processed input features
        """
        features = [
            race_conditions.get('track_position', 0),
            race_conditions.get('lap_number', 0),
            race_conditions.get('tire_wear', 1.0),
            race_conditions.get('fuel_level', 100),
            race_conditions.get('gap_to_leader', 0),
            # Add more race and driver-specific features
        ]
        
        return np.array([features])
    
    def simulate_driver_behavior(self, initial_conditions: Dict, total_laps: int = 60) -> List[Dict]:
        """
        Simulate driver behavior throughout the race
        
        Args:
            initial_conditions (Dict): Starting race conditions
            total_laps (int): Total number of laps
        
        Returns:
            List[Dict]: Behavior progression
        """
        behavior_progression = []
        current_conditions = initial_conditions.copy()
        
        for lap in range(1, total_laps + 1):
            current_conditions['lap_number'] = lap
            
            driver_action = self.predict_driver_actions(current_conditions)
            
            behavior_progression.append({
                'lap': lap,
                'predicted_action': driver_action['predicted_action'],
                'action_confidences': driver_action['action_confidences'],
                'conditions': current_conditions
            })
            
            # Update conditions for next lap
            current_conditions = self._update_behavior_conditions(current_conditions, driver_action)
        
        return behavior_progression
    
    def _update_behavior_conditions(self, current_conditions: Dict, driver_action: Dict) -> Dict:
        """
        Update race conditions based on predicted driver behavior
        
        Args:
            current_conditions (Dict): Current race conditions
            driver_action (Dict): Predicted driver action
        
        Returns:
            Dict: Updated race conditions
        """
        updated_conditions = current_conditions.copy()
        
        # Adjust conditions based on predicted action
        if driver_action['predicted_action'] == 'aggressive_overtake':
            updated_conditions['track_position'] -= 1  # Move up positions
        elif driver_action['predicted_action'] == 'defensive_driving':
            updated_conditions['gap_to_leader'] += 0.5  # Slow down slightly
        
        return updated_conditions
