import argparse
import sys
import traceback

from flask import Flask, request
from predictionService.powerGrid.LSTM_Prediction import LSTM_Prediction_Service

app = Flask(__name__)
# for storing service
service_dict = {}


@app.route("/")
def hello():
    return {"msg": "This is BTS Plain Server that you can use to test"}


# command: creating/removing service
@app.route("/command", methods=["POST"])
def execute_command():
    """
    Command schema:
    {
        "client_info": {
            "id": "aaltosea1",
            "roles": "ml_provider"
        },
        "model": {
            "name": "LSTM"
        },
        "command": "CREATE"
    }
    """
    try:
        req_data = request.get_json()
        # TO DO:
        # Check client info

        # Command processing
        if req_data["command"] is not None:
            command = req_data["command"]
            # Creat new service
            if command == "CREATE":
                if req_data["model"] is not None:
                    model_name = req_data["model"]["name"]
                    if model_name not in service_dict.keys():
                        if req_data["model"]["name"] == "LSTM":
                            # Create LSTM prediction service
                            req_service = LSTM_Prediction_Service()
                            service_dict["LSTM"] = req_service
                            status = f"{model_name} was created"
                        else:
                            status = f"{model_name} was not found"
                    else:
                        status = f"{model_name} has been already created"
                else:
                    status = "model must be specified"
            # Remove service
            elif command == "REMOVE":
                model_name = req_data["model"]["name"]
                if model_name in service_dict.keys():
                    service_dict.pop(model_name)
                    status = f"{model_name} was removed"
                else:
                    status = f"{model_name} was not found"
            else:
                status = f"{command} is not supported"
        else:
            status = "command is empty"

    # Exception tracing
    except Exception as e:
        print(f"[ERROR] - Error {type(e)} in execute_command: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())
        status = "error occurred"
    resp_info = {"status": status, "msg": f"the data is {req_data}"}
    return resp_info


# command: inference service
@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction request schema:

    {
        "model": "LSTM",
        "data":{ <@request_data> }
    }
    """
    try:
        req_data = request.get_json()
        if req_data["model"] is not None:
            model = req_data["model"]
            if model in service_dict:
                service = service_dict[model]
                print(service)
                if req_data["data"] is not None:
                    result = service.predict(req_data["data"])
                    status = "predict success"
                else:
                    status = "request data not found"
                    result = "False"
            else:
                status = "request model not found"
                result = "False"
        else:
            status = "model name not found"
            result = "False"
    # Exception tracing
    except Exception as e:
        print(f"[ERROR] - Error {type(e)} in execute_command: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())
        status = "error occurred"

    # Response
    resp_info = {
        "status": status,
        "msg": f"the data is {req_data}",
        "result": result,
    }
    return resp_info


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="Port of the service")

    args = parser.parse_args()
    port = int(args.port)
    app.run(debug=True, port=port)
