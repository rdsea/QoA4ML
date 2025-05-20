from qoa4ml.connector.amqp_connector import Amqp_Connector
import qoa4ml.utils as qoa_utils
import time, uuid, json

connetor_conf = qoa_utils.load_config("./conf/connector.json")["amqp_connector"]["conf"]

connector = Amqp_Connector(connetor_conf)
for i in range(2):
    body = json.dumps({"predictor": i, "result": "ready"})
    corr_id = "93ca40c5-7058-4281-86e3-f139733788fa"
    routing_key = "object_detection.notification." + corr_id
    connector.send_data(body, corr_id, routing_key=routing_key)
    print("result sent")
    time.sleep(1)
