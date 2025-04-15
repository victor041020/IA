
import sqlite3
import pickle
import numpy as np
from datetime import datetime

class DatabaseHandler:
    def __init__(self, db_name='ml_predictions.db'):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_binary BLOB
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    actual_value TEXT,
                    predicted_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(id)
                )
            ''')
            conn.commit()
    
    def save_model(self, name, model_type, model, metrics):
        model_binary = pickle.dumps(model)
        metrics_str = str(metrics)
        
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO models (name, type, metrics, model_binary) VALUES (?, ?, ?, ?)',
                (name, model_type, metrics_str, model_binary)
            )
            conn.commit()
            return cursor.lastrowid
    
    def save_predictions(self, model_id, y_true, y_pred):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            for actual, predicted in zip(y_true, y_pred):
                cursor.execute(
                    'INSERT INTO predictions (model_id, actual_value, predicted_value) VALUES (?, ?, ?)',
                    (model_id, str(actual), str(predicted))
                )
            conn.commit()
    
    def get_model(self, model_id):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT model_binary FROM models WHERE id = ?', (model_id,))
            result = cursor.fetchone()
            if result:
                return pickle.loads(result[0])
            return None
    
    def get_all_models(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, type, metrics, created_at FROM models')
            return cursor.fetchall()
