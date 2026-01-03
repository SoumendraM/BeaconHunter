import pandas as pd

def create_derived_features(beacon_df):
    # create derived feature based on variance of inter_event_seconds per host_id, dst_port
    beacon_df['beaconness'] = beacon_df.groupby('host_id')['inter_event_seconds'].transform(lambda x: x.var()).fillna(0)

    # create derived feature based on dst_port common vs rare ports
    common_ports = [80, 443, 53, 8080, 8443, 993, 995]
    beacon_df['wierdness'] = beacon_df['dst_port'].apply(lambda x: 'common' if x in common_ports else 'rare')

    # Create derived feature based on risk of process names 
    high_risk_processes = ['cmd.exe', 'cscript.exe', 'meterpreter.exe', 'mshta.exe', 'powershell.exe', 
                           'regsvr32.exe', 'rundll32.exe', 'sliver-client.exe', 'unknown.bin', 'wscript.exe']
    beacon_df['proc_risk'] = beacon_df['proc_name'].apply(lambda x: 'high' if x in high_risk_processes else 'low')

    country_risk_mapping = {
    "BR": 1,
    "CN": 1,
    "HK": 1,
    "IR": 1,
    "KP": 1,
    "NG": 1,
    "RU": 1,
    "TR": 1,
    "UA": 1,
    "VN":  1
    }
    beacon_df['geoip_risk'] = beacon_df['country_code'].map(country_risk_mapping).fillna(0)

    return beacon_df


