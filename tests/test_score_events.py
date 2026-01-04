import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
from score_events import main

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))



class TestScoreEvents(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.test_dir, 'test_input.csv')
        self.output_file = os.path.join(self.test_dir, 'test_output.csv')
        
        # Create minimal test data
        test_data = {
            'event_id': ['EVT1', 'EVT2', 'EVT3', 'EVT4', 'EVT5', 'EVT6', 'EVT7', 'EVT8', 'EVT9', 'EVT10', 'EVT11', 'EVT12', 'EVT13'],
            'host_id': ['host21', 'host21', 'host34', 'host45', 'host45', 'host22', 'host22', 'host22', 'host45', 'host22', 'host21', 'host22', 'host45'],
            'src_ip': ['1.1.1.1', '2.2.2.2', '3.3.3.3', '4.4.4.4', '5.5.5.5', '1.1.1.1', '2.2.2.2', '3.3.3.3', '4.4.4.4', '5.5.5.5', '1.1.1.1', '2.2.2.2', '3.3.3.3'],
            'dst_ip': ['1.1.1.1', '2.2.2.2', '3.3.3.3', '4.4.4.4', '5.5.5.5', '1.1.1.1', '2.2.2.2', '3.3.3.3', '4.4.4.4', '5.5.5.5', '1.1.1.1', '2.2.2.2', '3.3.3.3'],
            'signed_binary': [1, 1, 1, 1, 1, 2, 4, 4, 6, 55, 67, 11, 12],
            'protocol': ['tcp', 'http', 'https', 'dns', 'udp', 'tcp', 'http', 'https', 'dns', 'udp', 'https', 'dns', 'udp'],
            'user': ['user1', 'user2', 'user3', 'user4', 'user5', 'user1', 'user2', 'user3', 'user4', 'user5', 'user1', 'user2', 'user3'],
            'timestamp': pd.date_range('2024-01-01', periods=13),
            'inter_event_seconds': [553, 60, 991, 73, 52, 553, 60, 991, 73, 5, 553, 60, 991],
            'dst_port': [80, 8080, 53, 221, 90, 80, 8080, 53, 221, 90, 80, 8080, 53],
            'bytes_in': [222,333,444,555,666,222,333,444,555,666,222,333,444],
            'bytes_out': [111, 222, 333, 444, 555, 222,333,444,555,666, 222,333,444], 
            'proc_name': ['abc.exe', 'fre.exe', 'fire.exe', 'svchost.exe', 'rrc.exe', 'abc.exe', 'fre.exe', 'fire.exe', 'svchost.exe', 'rrc.exe', 'abc.exe', 'fre.exe', 'fire.exe'],
            'country_code': ['GT', 'YT', 'BR', 'OI', 'MN', 'GT', 'YT', 'BR', 'OI', 'MN', 'GT', 'YT', 'BR'],
            'label': [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
        }
        pd.DataFrame(test_data).to_csv(self.input_file, index=False)
    
    def tearDown(self):
        for file in [self.input_file, self.output_file]:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(self.test_dir)
    
    @patch('src.score_events.RandomForestModel')
    @patch('src.score_events.joblib.load')
    @patch('src.score_events.process_features')
    def test_output_csv_structure(self, mock_process, mock_joblib, mock_rf_model):
        mock_process.return_value = pd.DataFrame(np.random.rand(3, 3))
        mock_rf_instance = MagicMock()
        mock_rf_instance.risk_scores.return_value = np.array([0.3, 0.7, 0.2, 0.3, 0.7, 0.2, 0.3, 0.7, 0.2, 0.3, 0.7, 0.2, 0.9])
        mock_rf_model.return_value = mock_rf_instance
        
        mock_isolation = MagicMock()
        mock_isolation.decision_function.return_value = np.array([0.1, 0.8, 0.15, 0.1, 0.8, 0.15, 0.1, 0.8, 0.15, 0.1, 0.8, 0.15, 0.22])
        mock_joblib.return_value = mock_isolation
        
        with patch('sys.argv', ['score_events.py', '--input', self.input_file, '--output', self.output_file]):
            main()
        
        output_df = pd.read_csv(self.output_file)
        required_cols = ['event_id', 'host_id', 'risk_score', 'anomaly_score', 'fusion_risk_score', 'risk_label']
        self.assertEqual(list(output_df.columns), required_cols)
        self.assertEqual(len(output_df), 13)
    
    @patch('src.score_events.RandomForestModel')
    @patch('src.score_events.joblib.load')
    @patch('src.score_events.process_features')
    def test_risk_labels_valid(self, mock_process, mock_joblib, mock_rf_model):
        mock_process.return_value = pd.DataFrame(np.random.rand(3, 3))
        mock_rf_instance = MagicMock()
        mock_rf_instance.risk_scores.return_value = np.array([0.2, 0.6, 0.9, 0.2, 0.6, 0.9, 0.2, 0.6, 0.9, 0.2, 0.6, 0.9, 0.8])
        mock_rf_model.return_value = mock_rf_instance
        
        mock_isolation = MagicMock()
        mock_isolation.decision_function.return_value = np.array([0.3, 0.5, 0.9, 0.3, 0.5, 0.9, 0.3, 0.5, 0.9, 0.3, 0.5, 0.9, 0.8])
        mock_joblib.return_value = mock_isolation
        
        with patch('sys.argv', ['score_events.py', '--input', self.input_file, '--output', self.output_file]):
            main()
        
        output_df = pd.read_csv(self.output_file)
        valid_labels = {'HIGH', 'MEDIUM', 'LOW'}
        self.assertTrue(output_df['risk_label'].isin(valid_labels).all())
    
    @patch('src.score_events.RandomForestModel')
    @patch('src.score_events.joblib.load')
    @patch('src.score_events.process_features')
    def test_scores_normalized(self, mock_process, mock_joblib, mock_rf_model):
        mock_process.return_value = pd.DataFrame(np.random.rand(3, 3))
        mock_rf_instance = MagicMock()
        mock_rf_instance.risk_scores.return_value = np.array([0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4,0.8])
        mock_rf_model.return_value = mock_rf_instance
        
        mock_isolation = MagicMock()
        mock_isolation.decision_function.return_value = np.array([-1.0, 2.0, 0.5, -1.0, 2.0, 0.5, -1.0, 2.0, 0.5, -1.0, 2.0, 0.5, 0.6])
        mock_joblib.return_value = mock_isolation
        
        with patch('sys.argv', ['score_events.py', '--input', self.input_file, '--output', self.output_file]):
            main()
        
        output_df = pd.read_csv(self.output_file)
        self.assertTrue((output_df['anomaly_score'] >= 0).all() and (output_df['anomaly_score'] <= 1).all())


if __name__ == '__main__':
    unittest.main()