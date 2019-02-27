from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

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

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
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
def index(request):
    html = path / 'view' / 'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    confidence = (round(max((learn.predict(img)[2]).tolist()),2) * 100)
    result = str(learn.predict(img)[0])
    return JSONResponse(
        {
            'result': result,
            'predictions': confidence,
        }
    )


if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='127.0.0.1', port=8080)
