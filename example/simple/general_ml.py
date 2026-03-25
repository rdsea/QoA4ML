import argparse
import random
import sys
import time
import traceback
from threading import Thread

from qoa4ml.qoa_client import QoaClient
from qoa4ml.reports.ml_report_model import GeneralMlInferenceReport
from qoa4ml.reports.ml_reports import MLReport

client1 = QoaClient(report_cls=MLReport, config_path="./config/client1.yaml")
client2 = QoaClient(report_cls=MLReport, config_path="./config/client2.yaml")
client3 = QoaClient(report_cls=MLReport, config_path="./config/client3.yaml")
client4 = QoaClient(report_cls=MLReport, config_path="./config/client4.yaml")
client5 = QoaClient(report_cls=MLReport, config_path="./config/client5.yaml")
# client5.process_monitor_start(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Node Monitoring")
    parser.add_argument("--th", help="Number of thread", default=1)
    parser.add_argument("--sl", help="Sleep time", default=-1)
    parser.add_argument("--client", help="Client config file", default="./conf.json")
    args = parser.parse_args()

    concurrent = int(args.th)
    time_sleep = float(args.sl)
    print("checkpoint 0")

    def sender(num_thread):
        count = 0
        error = 0
        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                # client1.timer()
                print("This is thread: ", num_thread, "Starting request: ", count)
                client1.observe_metric("metric1", random.randint(1, 100), 0)
                client1.observe_metric("metric2", random.randint(1, 100), 0)
                client1.observe_metric("image_width", random.randint(1, 100), 1)
                client1.observe_metric("image_height", random.randint(1, 100), 1)
                client1.observe_metric("object_width", random.randint(1, 100), 1)
                client1.observe_metric("object_height", random.randint(1, 100), 1)
                client1.timer()
                report_1 = client1.report()
                # print((report_1.model_dump_json(exclude_none=True)))

                # client2.import_previous_report(report_1)
                # print("Thread - ",num_thread, " Response1:", report_1)

                client2.import_previous_report(report_1)
                client2.timer()
                # print("This is thread: ",num_thread, "Starting request: ", count)
                client2.observe_metric("metric1", random.randint(1, 100), 0)
                client2.observe_metric("metric2", random.randint(1, 100), 0)
                client2.observe_metric("image_width", random.randint(1, 100), 1)
                client2.observe_metric("image_height", random.randint(1, 100), 1)
                client2.observe_metric("object_width", random.randint(1, 100), 1)
                client2.observe_metric("object_height", random.randint(1, 100), 1)
                client2.timer()
                report_2 = client2.report()
                # # print("Thread - ",num_thread, " Response2:", report_2)

                client3.import_previous_report(report_2)
                client3.timer()
                # print("This is thread: ", num_thread, "Starting request: ", count)
                client3.observe_inference(random.randint(1, 1000))
                client3.observe_inference_metric("confidence", random.randint(1, 100))
                client3.observe_inference_metric("accuracy", random.randint(1, 100))
                client3.timer()
                report_3 = client3.report()
                # debug(report_3)
                # print("Thread - ",num_thread, " Response3:", report_3)

                client4.import_previous_report(report_2)
                client4.timer()
                # print("This is thread: ", num_thread, "Starting request: ", count)
                client4.observe_inference(random.randint(1, 1000))
                client4.observe_inference_metric("confidence", random.randint(1, 100))
                client4.observe_inference_metric("accuracy", random.randint(1, 100))
                client4.timer()
                report_4 = client4.report()
                # debug(report_4)
                # print("Thread - ",num_thread, " Response4:", report_4)

                # debug(report_3.inference_report)
                # debug(report_4.inference_report)
                client5.import_previous_report([report_3, report_4])
                client5.timer()
                # print("This is thread: ", num_thread, "Starting request: ", count)
                client5.observe_inference(random.randint(1, 1000))
                client5.observe_inference_metric("confidence", random.randint(1, 100))
                client5.observe_inference_metric("accuracy", random.randint(1, 100))
                client5.timer()
                report_5 = client5.report(submit=True)
                # debug(report_5)
                with open(
                    "qoa4ml-general-ml-report-schema.json", "w", encoding="utf-8"
                ) as file:
                    file.write(GeneralMlInferenceReport(**report_5).model_dump_json())
                # print("Thread - ",num_thread, " Response5:", report_5)
            except Exception:
                error += 1
                # qoaLogger.error("Error {} in merge_dict: {}".format(type(e),e.__traceback__))
                traceback.print_exception(*sys.exc_info())
            count += 1
            if time_sleep == -1:
                time.sleep(1)
            else:
                time.sleep(time_sleep)

    for i in range(concurrent):
        t = Thread(target=sender, args=[i])
        t.start()
