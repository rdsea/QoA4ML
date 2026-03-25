import os

import numpy as np

from qoa4ml import probes as qoa_probes
from qoa4ml import qoaUtils as qoa_utils
from qoa4ml.qoa_client import Qoa_Client

from .ML_Loader import ML_Loader


class LSTM_Prediction_Service:
    def __init__(self):
        # init base directory
        self.base_dir = os.environ.get("LSTM_BASE_DIR")
        if not self.base_dir:
            print("LSTM_BASE_DIR is not defined")
            self.base_dir = qoa_utils.get_file_dir(__file__)
            print("LSTM_BASE_DIR is set to: ", self.base_dir)
        self.base_dir = str(self.base_dir)

        # Read service config by default
        self.config = qoa_utils.load_config(self.base_dir + "/config/lstm.json")
        print(self.config)

        # Init the queue for ML request and load the ML model
        self.model_info = self.config["model"]

        ############################################### INIT QOA CLIENT ##################################################################
        self.qoa_config = self.config["qoa_service"]
        self.qoa_client = Qoa_Client(
            self.qoa_config["client_info"], self.qoa_config["registration"]
        )
        ####################################################################################################################################

        # data normalize config
        self.normalize = self.config["data_normalize"]

        # Load ML model
        self.model = ML_Loader(self.model_info, self.base_dir)

    def ML_prediction(self, pas_series):
        # Making prediction using loader
        result = self.model.prediction(pas_series)
        result = result.reshape(result.shape[1], result.shape[2])
        # Load the result into json format
        data_js = {
            "LSTM": float(result[0]),
        }
        self.print_result(data_js)
        return data_js

    def predict(self, payload):
        # Start qoa timer for calculating response time
        self.qoa_client.timer()

        # Load json message
        norm_1 = float(payload["norm_1"])
        norm_2 = float(payload["norm_2"])
        norm_3 = float(payload["norm_3"])
        norm_4 = float(payload["norm_4"])
        norm_5 = float(payload["norm_5"])
        norm_6 = float(payload["norm_6"])
        pas_series = np.array(
            [[norm_1], [norm_2], [norm_3], [norm_4], [norm_5], [norm_6]]
        )
        pas_series = np.array(pas_series)[np.newaxis, :, :]

        # Call back the ML prediction server for making prediction
        response = self.ML_prediction(pas_series)
        payload["LSTM"] = response["LSTM"]

        ####################### SEND THE QOA4ML REPORT #########################
        # Response_time is already report by calling timer 2nd time
        self.qoa_client.timer()
        # Estimate data accuracy using qoa4ml library
        data_accuracy = qoa_probes.eva_none(pas_series)
        # Add data accuracy to QoA report
        self.qoa_client.observeMetric("data_accuracy", data_accuracy, data_quality=True)
        # Report prediction
        inference_id = self.qoa_client.inference_report(value=response["LSTM"])
        # Get QoA report - submit=True to submit report to QoA Observability server
        qoa_report = self.qoa_client.get_report(submit=True)
        print("--------------\nQoA Report: \n", qoa_report, "\n--------------")
        ########################################################################

        payload["inference_id"] = inference_id
        self.print_result(response)
        return payload

    def print_result(self, data):
        prediction = ""
        for key in data:
            prediction += f"\n# {key} : {data[key]} "

        prediction_to_str = f"""{"=" * 40}
        # Prediction Server:{prediction}
        {"=" * 40}"""
        print(prediction_to_str.replace("  ", ""))
