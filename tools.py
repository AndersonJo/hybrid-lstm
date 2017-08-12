import os
import urllib.request
import zipfile
import pandas as pd

from io import BytesIO, StringIO


def preprocess(data: pd.DataFrame):
    # Lower column names
    data.columns = map(str.lower, data.columns)

    # Datetime
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data.set_index('datetime', inplace=True)
    del data['date']
    del data['time']

    return data


def load_household_power_consumption(dest='dataset'):
    """
    https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

    1.date: Date in format dd/mm/yyyy
    2.time: time in format hh:mm:ss
    3.global_active_power: household global minute-averaged active power (in kilowatt)
    4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)
    5.voltage: minute-averaged voltage (in volt)
    6.global_intensity: household global minute-averaged current intensity (in ampere)
    7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
    8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
    9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.
    """
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
    ZIP_FILE_NAME = 'household_power_consumption.txt'
    CSV_PATH = os.path.join('dataset', 'household_power_consumption.csv')

    ##################################
    # Check existing file
    ##################################
    if not os.path.exists(dest):
        os.mkdir(dest)

    ##################################
    # Download and Unzip file
    ##################################
    if not os.path.exists(CSV_PATH):
        with urllib.request.urlopen(URL) as res:
            f = BytesIO(res.read())
            zip_ref = zipfile.ZipFile(f)
            data_txt = zip_ref.read(ZIP_FILE_NAME).decode('utf-8')
            zip_ref.close()

        data = pd.read_csv(StringIO(data_txt), sep=';')
        data = preprocess(data)
        data.to_csv(CSV_PATH)
        print('save')
    else:
        print('CSV!')
        data = pd.read_csv(CSV_PATH, index_col=0)

    print(data.head())


if __name__ == '__main__':
    load_household_power_consumption()
