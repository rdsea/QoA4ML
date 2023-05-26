import pymongo
from qoa4ml.collector.amqp_collector import Amqp_Collector
import qoa4ml.utils as utils
from threading import Thread, Timer
import time, json, os, uuid
import pandas as pd
import argparse, random
import traceback,sys,pathlib

from flask import Flask, jsonify, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


def get_dict_at(dict, i):
    keys = list(dict.keys())
    return dict[keys[i]], keys[i]

application_list = {}
agent_list = {}

class Rohe_Agent(object):
    def __init__(self, configuration):
        self.conf = configuration
        colletor_conf = self.conf["collector"]
        self.collector = Amqp_Collector(colletor_conf['amqp_collector']['conf'], host_object=self)
        db_conf = self.conf["database"]
        self.mongo_client = pymongo.MongoClient(db_conf["url"])
        self.db = self.mongo_client[db_conf["db_name"]]
        self.metric_collection = self.db[db_conf["metric_collection"]]

        self.insert_db = True
    
    def reset_db(self):
        self.metric_collection.drop()

    def start_consuming(self):
        print("Start Consuming")
        self.collector.start()

    def start(self):
        sub_thread = Thread(target=self.start_consuming)
        sub_thread.start()
        print("start consumming message")



    def message_processing(self, ch, method, props, body):
        mess = json.loads(str(body.decode("utf-8")))
        print("Receive QoA Report: \n", mess)
        if self.insert_db:
            insert_id = self.metric_collection.insert_one(mess)
            print("Insert to database", insert_id)

    def stop(self):
        # self.collector.stop()
        self.insert_db = False
    def restart(self):
        # self.collector.stop()
        self.insert_db = True
    


class Rohe_ObService(Resource):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.conf = kwargs
        self.db_config = self.conf["database"]
        self.connector_config = self.conf["connector"]
        self.collector_config = self.conf["collector"]
        
        
    def get(self):
        args = request.query_string.decode("utf-8").split("&")
        # get param from args here
        return jsonify({'status': args})

    def post(self):
        if request.is_json:
            args = request.get_json(force=True)
            print(args)
            response = {}
            if "application" in args:
                application_name = args["application"]
                if application_name not in application_list:
                    application_list[application_name] = {}
                    application_list[application_name]["id"] = str(uuid.uuid4())
                    application_list[application_name]["client_count"] = 0
                    response[application_name] = "Application {} created".format(application_name)
                else:
                    response[application_name] = "OK"
                    
                application_list[application_name]["client_count"] += 1
                # TO DO
                # Check client_id, role, stage_id, instance_name

                # Prepare connector for QoA Client
                connector = self.connector_config.copy()
                for key in list(connector.keys()):
                    connector_i = connector[key]
                    i_config = connector_i["conf"]
                    i_config["exchange_name"] = str(application_name)+"_exchange"
                    i_config["out_routing_key"] = str(application_name)
                    if "client_id" in args:
                        i_config["out_routing_key"] = i_config["out_routing_key"]+"."+args["client_id"]
                    if "stage_id" in args:
                        i_config["out_routing_key"] = i_config["out_routing_key"]+"."+args["stage_id"]
                    if "instance_name" in args:
                        i_config["out_routing_key"] = i_config["out_routing_key"]+"."+args["instance_name"]
                    i_config["out_routing_key"] = i_config["out_routing_key"]+".client"+str(application_list[application_name]["client_count"])
                response["application_id"] = application_list[application_name]["id"]
                response["connector"] = connector
                
                # Prepare QoA Agent
                application_id = application_list[application_name]["id"]
                if application_id not in agent_list:
                    # Database configuration
                    agent_db_config = self.db_config.copy()
                    agent_db_config["db_name"] = application_name+"_"+application_id
                    agent_db_config["metric_collection"] = "metric_collection"
                    # Collector configuration
                    collector_config = self.collector_config.copy()
                    for key in list(collector_config.keys()):
                        collector_i = collector_config[key]
                        i_config = collector_i["conf"]
                        i_config["exchange_name"] = str(application_name)+"_exchange"
                        i_config["out_routing_key"] = str(application_name)+".#"
                    agent_config ={}
                    agent_config["database"] = agent_db_config
                    agent_config["collector"] = collector_config
                    agent = Rohe_Agent(agent_config)
                    agent_id = str(uuid.uuid4())
                    agent_list[application_id] = {}
                    agent_list[application_id][agent_id] = {}
                    agent_list[application_id][agent_id]["agent"] = agent
                    agent_list[application_id][agent_id]["status"] = "stop"
                    agent_list[application_id][agent_id]["configuration"] = agent_config
                    print(agent_list)
                    agent_list[application_id][agent_id]["agent"].start()

            else:
                response["Error"] = "Application name not found"
            
           
            
        return jsonify({'status': "success", "response":response})

    def put(self):
        if request.is_json:
            args = request.get_json(force=True)
        # get param from args here
        return jsonify({'status': True})

    def delete(self):
        if request.is_json:
            args = request.get_json(force=True)
        # get param from args here
        return jsonify({'status': args})


if __name__ == '__main__': 
    # init_env_variables()
    parser = argparse.ArgumentParser(description="Argument for Rohe Observation Service")
    parser.add_argument('--conf', help='configuration file', default=None)
    args = parser.parse_args()
    config_file = args.conf
    if not config_file:
        config_file = utils.get_parent_dir(__file__,2)+"/config/rohe_ObS/conf.json"
        print(config_file)
    configuration = utils.load_config(config_file)

    api.add_resource(Rohe_ObService, '/registration',resource_class_kwargs=configuration)
    app.run(debug=True, port=5001)