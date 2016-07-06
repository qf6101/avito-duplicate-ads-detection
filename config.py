import json
import os
dir_here = os.path.dirname(os.path.realpath(__file__))
config = json.load(open(os.path.join(dir_here, 'config.json')))