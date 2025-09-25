import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from sklearn.linear_model import LogisticRegression


def id_residential_properties(properties_df, parcel_df):
    """ Subsets data for only R2 residential properties
    Args:
        properties_df (pd.DataFrame): contains the city property data
        parcel_df (pd.DataFrame): contains the parcel number and shape information

        Returns:
            pd.DataFrame:
                table with R2 properties and their parcel information
    """
    # format data
    properties_df['Number of Units'] = properties_df['Number of Units'].replace(',', '')
    properties_df['Number of Units'] = pd.to_numeric(properties_df['Number of Units'], errors='coerce')

    # R2 classifications
    residence_types = ['Multi-Family Residential', 'Commercial Hotel']

    residential_properties = properties_df[properties_df['Use Definition'].isin(residence_types)]
    residential_properties = residential_properties[
        ['Property Location', 'Parcel Number', 'Block', 'Lot', 'Use Code', 'Use Definition', 'Property Class Code',
         'Property Class Code Definition', 'Year Property Built', 'Number of Stories', 'Number of Units',
         'the_geom']].drop_duplicates(subset=['Parcel Number', 'Lot'])

    low_units = ['Apartment 4 units or less', 'Flat & Store 4 units or less', 'Dwellings - Apt 4 units or less',
                 'TIC Bldg 4 units or less', '2 Dwellings on 1 Parcel 4 units or less',
                 'Office and Apartments 4 units or less', 'Apartment Gov 4 units or less',
                 "Single Struct on Multi Lot(D & F's only)"]

    # keep any properties specifying 3 or more units
    r2_properties = residential_properties[np.logical_or(residential_properties['Number of Units'] >= 3,
                                                         residential_properties['Number of Units'].isna())]
    # if unknown unit number, drop and low unit properties
    r2_properties = r2_properties[~np.logical_and(r2_properties['Number of Units'].isna(),
                                                  r2_properties['Property Class Code Definition'].isin(low_units))]
    r2_properties = r2_properties.reset_index(drop=True)

    # merge in parcel data
    parcel_data = parcel_df.rename(columns={'blklot': 'Parcel Number'})
    r2_properties_parcel = r2_properties.merge(parcel_data[['Parcel Number', 'shape']], on='Parcel Number', how='left')
    r2_properties_parcel = r2_properties_parcel[~r2_properties_parcel['shape'].isna()]

    return r2_properties_parcel


def map_points_to_parcel(r2_properties_parcel, points_df):
    """ Maps points to the polygon (parcel) it is contained in
    Args:
        r2_properties_parcel (pd.DataFrame): contains the R2 parcel data
        points_df (pd.DataFrame): contains the point and address name

        Returns:
            pd.DataFrame:
                table with points, parcel, and address information
    """

    geometries = r2_properties_parcel['shape'].apply(wkt.loads)
    parcel_polys = gpd.GeoDataFrame(
        {"Parcel Number": r2_properties_parcel['Parcel Number']},
        geometry=geometries,
        crs="EPSG:4326"
    )

    points = points_df['Location'].apply(wkt.loads)
    fire_points = gpd.GeoDataFrame(
        {"Address": points_df['Address']},
        geometry=points,
        crs="EPSG:4326"
    )

    # assign each point to the polygon it falls inside
    point_parcel_map = gpd.sjoin(fire_points, parcel_polys, how="inner", predicate="within").reset_index(drop=True)
    point_parcel_map = point_parcel_map.rename(columns={'geometry': 'Location'})
    point_parcel_map = point_parcel_map.drop(columns='index_right')

    return point_parcel_map


def compute_fire_incident_stats(fire_incidents, alarm_thresh=1):
    """ Generate stats about low/high alarm incidents and injury data
    Args:
        fire_incidents (pd.DataFrame): contains data on fire incidents
        alarm_thresh (int): max threshold for low alarm fir incident, defaults to 1

        Returns:
            pd.DataFrame:
                table with points, parcel, and address information
    """

    # create new column Low Alarm Incident, set to False if incident was more than specified alarm fire classification
    fire_incidents['Low Alarm Incident'] = True
    fire_incidents.loc[fire_incidents['Number of Alarms'] > alarm_thresh, 'Low Alarm Incident'] = False

    # sum the number of fire related injuries
    fire_incident_injury = fire_incidents[['Location', 'Fire Fatalities', 'Fire Injuries']].groupby(
        by=['Location']).sum().reset_index()

    # sum the number of low alarm and high alarm incidents
    fire_incidents_low_alarm = fire_incidents[['Location', 'Low Alarm Incident', 'ID']].groupby(
        by=['Location', 'Low Alarm Incident']).count()
    fire_incidents_low_alarm = fire_incidents_low_alarm.reset_index().rename(columns={'ID': 'count'})

    fire_incidents_low_alarm = fire_incidents_low_alarm.sort_values(by='Low Alarm Incident').pivot(
        index='Location', columns='Low Alarm Incident', values='count').reset_index()
    fire_incidents_low_alarm = fire_incidents_low_alarm.rename(columns={False: 'High Alarm Incidents',
                                                                        True: 'Low Alarm Incidents'})
    fire_incidents_low_alarm.fillna(0, inplace=True)

    # combine the two data frames
    fire_incident_stats = pd.merge(fire_incident_injury, fire_incidents_low_alarm, on='Location')

    return fire_incident_stats


def compute_fire_violation_stats(fire_violations):
    """ Generate stats about number of fire code violations
    Args:
        fire_violations (pd.DataFrame): contains data on fire code violations

        Returns:
            pd.DataFrame:
                table with data on number of total and open violations
    """

    # total violations
    fire_violations_risk = fire_violations[['Location', 'Violation Number']].groupby(by=['Location']).count()

    # open violations
    fire_violations_open = fire_violations[fire_violations.Status == 'open']
    fire_violations_open = fire_violations_open[['Location', 'Violation Number']].groupby(by=['Location']).count()
    fire_violations_open = fire_violations_open.rename(columns={'Violation Number': 'Open Violation Number'})

    # combine the two data frames
    fire_violations_stats = pd.merge(fire_violations_risk, fire_violations_open, on='Location')

    return fire_violations_stats


def assign_risk_scores(fire_data):
    """ Uses mean and 99th percentile to calculate a risk score for each property
    Args:
        fire_data (pd.DataFrame): contains data with all generated property features

        Returns:
            pd.DataFrame:
                table risk_score column appended
    """

    fire_data['risk_score'] = 0
    fire_data.loc[fire_data['High Alarm Incidents'] > fire_data['High Alarm Incidents'].mean(), 'risk_score'] += 2

    fire_data.loc[fire_data['Low Alarm Incidents'] > fire_data['Low Alarm Incidents'].mean(), 'risk_score'] += 1
    fire_data.loc[
        fire_data['Low Alarm Incidents'] > np.percentile(fire_data['Low Alarm Incidents'], 99), 'risk_score'] += 2

    fire_data.loc[fire_data['Open Violation Number'] > fire_data['Open Violation Number'].mean(), 'risk_score'] += 1
    fire_data.loc[
        fire_data['Open Violation Number'] > np.percentile(fire_data['Open Violation Number'], 99.9), 'risk_score'] += 2

    fire_data.loc[fire_data['Number of Stories'] > fire_data['Number of Stories'].mean(), 'risk_score'] += 1

    fire_data.loc[fire_data['Number of Units'] > fire_data['Number of Units'].mean(), 'risk_score'] += 1

    fire_data.loc[fire_data['Year Property Built'] < fire_data['Year Property Built'].mean(), 'risk_score'] += 1

    return fire_data


def classify_property_violations(df, target_var, features, curr_year=2024, threshold=None):
    """ Uses mean and 99th percentile to calculate a risk score for each property
    Args:
        df (pd.DataFrame): contains the feature data
        target_var (str): the column we will be predicting, must be binary
        features (list): list of columns to use as features
        curr_year (int): year to test data on, defaults to 2024
        threshold (float): new cutoff for classification, defaults to None (0.5)

        Returns:
            pd.Series:
                the actual target values, the predicted target values, and the associated probabilities
    """

    # train-test split by year (train on 2019-2023, test on 2024)
    train_data = df[df["Year"] < curr_year]
    test_data = df[df["Year"] == curr_year]

    X_train = train_data[features].fillna(0)
    y_train = train_data[target_var]
    X_test = test_data[features].fillna(0)
    y_test = test_data[target_var]

    # fit logistic regression
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train, y_train)

    # predict
    y_pred = log_reg.predict(X_test)
    y_proba = log_reg.predict_proba(X_test)[:, 1]

    # adjust threshold if specified
    if threshold:
        y_pred = (y_proba >= threshold).astype(int)

    return log_reg, y_test, y_pred, y_proba


def retrieve_recent_inspection(fire_inspections, compare_date=pd.Timestamp.today()):
    recent_inspections = fire_inspections[['Location', 'Inspection Start Date']]
    recent_inspections['Inspection Start Date'] = pd.to_datetime(recent_inspections['Inspection Start Date'])
    recent_inspections = recent_inspections.sort_values('Inspection Start Date', ascending=False).drop_duplicates(
        'Location', keep='first')
    recent_inspections = recent_inspections.dropna()
    recent_inspections['date_since_inspection'] = compare_date - recent_inspections['Inspection Start Date']

    return recent_inspections
