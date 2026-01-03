import pandas as pd
import pytest
from src.features import create_derived_features

def test_beaconness_feature():
    df = pd.DataFrame({
        'host_id': ['host1', 'host1', 'host1', 'host2', 'host2'],
        'inter_event_seconds': [10, 10, 10, 5, 15],
        'dst_port': [80, 80, 80, 443, 443],
        'proc_name': ['chrome.exe', 'chrome.exe', 'chrome.exe', 'firefox.exe', 'firefox.exe'],
        'country_code': ['US', 'US', 'US', 'US', 'US']
    })
    result = create_derived_features(df)
    assert 'beaconness' in result.columns
    assert result[result['host_id'] == 'host1']['beaconness'].iloc[0] == 0.0


def test_wierdness_feature():
    df = pd.DataFrame({
        'host_id': ['host1', 'host1', 'host1'],
        'inter_event_seconds': [10, 10, 10],
        'dst_port': [80, 443, 12345],
        'proc_name': ['chrome.exe', 'chrome.exe', 'chrome.exe'],
        'country_code': ['US', 'US', 'US']
    })
    result = create_derived_features(df)
    assert result.loc[0, 'wierdness'] == 'common'
    assert result.loc[2, 'wierdness'] == 'rare'


def test_proc_risk_feature():
    df = pd.DataFrame({
        'host_id': ['host1', 'host1', 'host1'],
        'inter_event_seconds': [10, 10, 10],
        'dst_port': [80, 80, 80],
        'proc_name': ['powershell.exe', 'notepad.exe', 'cmd.exe'],
        'country_code': ['US', 'US', 'US']
    })
    result = create_derived_features(df)
    assert result.loc[0, 'proc_risk'] == 'high'
    assert result.loc[1, 'proc_risk'] == 'low'
    assert result.loc[2, 'proc_risk'] == 'high'


def test_geoip_risk_feature():
    df = pd.DataFrame({
        'host_id': ['host1', 'host1', 'host1'],
        'inter_event_seconds': [10, 10, 10],
        'dst_port': [80, 80, 80],
        'proc_name': ['chrome.exe', 'chrome.exe', 'chrome.exe'],
        'country_code': ['CN', 'RU', 'US']
    })
    result = create_derived_features(df)
    assert result.loc[0, 'geoip_risk'] == 1
    assert result.loc[1, 'geoip_risk'] == 1
    assert result.loc[2, 'geoip_risk'] == 0