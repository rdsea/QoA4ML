# An Implementation of QoA4ML Observability based on OpenPolicyAgent (OPA)

In this package, we explain one implementation of the QoA4ML Observability using OPA. [OPA](https://www.openpolicyagent.org) is a general policy engine. We use it so that the QoA4ML can also be combined with other types of policies.

## Current development

Note that at the moment, the supporting tool is limited. Therefore, mostly the developer has to write [contracts and policies](../../language/).

### Develop Contracts and Policies

The contract  and policy should be ideally separated into json specification and Rego code. The developer can combine qoa4ml aspects with other types of policies

### Runtime metrics

Runtime metrics captured from monitoring system should be sent for checking violation. How to capture is described in the monitoring. To send metrics for checking, metrics must be in the right format and sent to the OPA service for QoA4ML through REST APIs

## Setup an OPA service for QoA4ML

Run a simple OPA server:

```
$docker pull openpolicyagent/opa:latest
$docker run -p 8181:8181 openpolicyagent/opa:latest run --server --set=decision_logs.console=true
```
## Samples of contracts and core QoA4ML terms

You can find [samples of contracts](../../language/qoa4mlopa/); the core QoA4ML terms are in [the language directory](../../language/). Worked examples live alongside this README in the per-scenario directories (`bts_example/`, `malware_detection/`, `object_detection/`).

## Add a contract

Here are few examples for testings. Note that you can look at [the OPA APIs for understanding how to the APIs](https://www.openpolicyagent.org/docs/latest)

the root path for storing documents (in our case: contracts) is **v1/data**.
Contracts can be stored in sub paths.

### Contract

Contracts are documents so we use OPA data APIs to store and manage them. Let us assume that we want to store a contract **bts-ex1** into a path **qoa4ml/bts/alarm**.

Add a contract

```bash
curl --location --request PUT 'localhost:8181/v1/data/qoa4ml/bts/alarm/bts-contract-ex1' \
--header 'Content-Type: application/json' \
--data-raw '{ DATA}'
```

Read a contract

```bash
curl --location --request GET 'localhost:8181/v1/data/qoa4ml/bts/alarm/bts-contract-ex1'
```

### Policies
Policies are used to check if a contract is violated. Policies are Rego code. We use policies APIs to store Policies. Let us assume that we have a **bts-policies-ex1**

add a policies:

```bash
curl --location --request PUT 'localhost:8181/v1/policies/bts-policies-ex1' \
--header 'Content-Type: text/plain' \
--data-raw ' DATA '
```
Get a policy:

```bash
curl --location --request GET 'localhost:8181/v1/policies'
```

**In the current prototype: we also put policies and contracts into a single file to speed up the prototype. but in principle they should be separated**

### Check if a policy/contract is violated

To check if a contract is violated, an ML observability service will send an input of runtime metrics according to the pre-defined specification and indicate the policy for such runtime metrics. We use OPA Evaluation API to check it. For example, to check if **INPUT** is valid for policies specified in  **bts-policies-ex1** we can have

```bash
url --location --request POST 'localhost:8181/v1/data/qoa4ml/bts/alarm/contract_violation' \
--header 'Content-Type: application/json' \
--data-raw '{"input":INPUT}'
```

where as:
* qoa4ml/bts/alarm/contract_violation: is the id of the policy (based on the package name and the decision specification in the code)
* INPUT is the json input of runtime metrics


## A fast test

* install the service
* Put the embedded policy:

```bash
curl --location --request PUT 'localhost:8181/v1/policies/bim-policies-basic-embedded' \
--header 'Content-Type: text/plain' \
--data-raw 'POLICIES'
```

where POLICIES is the content of **language/qoa4mlopa/bim-policies-basic-embedded.rego**

* Run the python code for evaluation: for example, if you put the [bim-policies-basic-embeeded.rego] into the service and play
```
$python3 simple_evaluation.py --purl http://localhost:8181/v1/data/qoa4ml/bim/basic/embedded/mlaccuracy_violation --input bim-runtimemetrics-ex1.json
b'{"decision_id":"4d4bc408-959e-427b-8031-6f2a0754066d","result":[true]}'
truong@aaltosea:~/myprojects/mygit/aaltosea/qoa4ml/observability/qoa4mlopa/src$ python3 simple_evaluation.py --purl http://localhost:8181/v1/data/qoa4ml/bim/basic/embedded/mlfair_violation --input bim-runtimemetrics-ex1.json
b'{"decision_id":"5d99f18a-f02b-42ec-9a86-4dc3d8856344","result":[false]}'
```
