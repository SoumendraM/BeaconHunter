import sys
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from src.train_detector import main

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_beacon_data():
    data = {
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
    df = pd.DataFrame(data)
    df.set_index(['event_id', 'host_id'], inplace=True)
    return df


@pytest.fixture
def temp_csv(sample_beacon_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_beacon_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


def test_main_runs_with_small_dataset(temp_csv):
    ## Sanity test that train_detector.py can run with a minimal dataset.
    with patch('pandas.read_csv', return_value=pd.read_csv(temp_csv)):
        with patch('src.train_detector.train_supervised_models') as mock_supervised:
            with patch('src.train_detector.train_unsupervised_model') as mock_unsupervised:
                with patch('src.train_detector.calculate_fusion_risk_scores') as mock_fusion:
                    mock_supervised.return_value = (MagicMock(), [0.5] * 13)
                    mock_unsupervised.return_value = (MagicMock(), [0] * 13, [0.3] * 13)
                    mock_fusion.return_value = [0.4] * 13
                    
                    main()
                    
                    assert mock_supervised.called
                    assert mock_unsupervised.called
                    assert mock_fusion.called