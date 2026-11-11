"""Serve the local Bayesian network explorer using the standard library."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import prob_inf

ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'web'
MODELS = {'burglary': 'burglary_alarm.csv', 'student': 'student_bn.csv'}


def load_model(name):
    if name not in MODELS:
        raise ValueError('Unknown network.')
    return prob_inf.read_model_file(ROOT / MODELS[name])


def describe_model(model):
    nodes = []
    for name in model.vars:
        parents = sorted(model.varDist[name].parents)
        rows = []
        for values in prob_inf.truefalse_combination_iterator(parents):
            rows.append({'values': values, 'probability': prob_inf.read_cpt(model, name, values)})
        nodes.append({'id': name, 'parents': parents, 'rows': rows})
    return {'nodes': nodes}


def calculate(data):
    if not isinstance(data, dict):
        raise ValueError('Request must be an object.')
    model = load_model(data.get('model'))
    query = data.get('query')
    value = data.get('value')
    evidence = data.get('evidence')
    method = data.get('method')
    if query not in model.vars or type(value) is not bool:
        raise ValueError('Choose a valid query and value.')
    if not isinstance(evidence, dict) or any(k not in model.vars or type(v) is not bool for k, v in evidence.items()):
        raise ValueError('Evidence must contain valid variables and Boolean values.')
    if query in evidence:
        raise ValueError('The query variable cannot also be evidence.')
    methods = {'tree': prob_inf.calc_query_exact_tree, 'brute': prob_inf.calc_query_exact_brute}
    if method not in methods:
        raise ValueError('Choose a valid inference method.')
    try:
        probability = methods[method](model, query, value, evidence)
    except ZeroDivisionError as exc:
        raise ValueError('This evidence has zero probability, so the posterior is undefined.') from exc
    return {'probability': probability}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def json_response(self, status, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/api/model/'):
            try:
                self.json_response(200, describe_model(load_model(self.path.removeprefix('/api/model/'))))
            except ValueError as exc:
                self.json_response(400, {'error': str(exc)})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/api/infer':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 16384:
                raise ValueError('Invalid request size.')
            data = json.loads(self.rfile.read(length))
            self.json_response(200, calculate(data))
        except (ValueError, TypeError) as exc:
            self.json_response(400, {'error': str(exc)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Bayesian network explorer locally')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping explorer.')
    finally:
        server.server_close()
