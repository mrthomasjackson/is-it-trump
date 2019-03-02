# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging
import aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

from flask import Flask
from flask_cors import CORS, cross_origin

model_file_url = 'https://www.dropbox.com/s/l6o5tp73n4uzujl/stage-2.pth?dl=1'
model_file_name = 'model'
classes = [
    'Abraham Lincoln',
    'Adolf Hitler',
    'Barack Obama',
    'Condoleezza Rice',
    'Dick Cheny',
    'Donald Trump',
    'George HW Bush',
    'George W Bush',
    'Hillary Clinton',
    'Jimmy Carter',
    'John F Kennedy',
    'Martin Luther King',
    'Nancy Pelosi',
    'Richard Nixon',
    'Ronald Regan'
]
path = Path(__file__).parent


app = Flask(__name__, static_url_path='static')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=None) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner():
    await download_file(model_file_url, path / 'models' / f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
                                                    ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet152, pretrained=False)
    learn.load(model_file_name)
    return learn


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
@cross_origin()
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'


@app.errorhandler(500)
@cross_origin()
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]
