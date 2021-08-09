from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
from Poc1WithTop2 import testmodel


app = FastAPI()

@app.get('/',response_class=HTMLResponse)
def basic_view():
    return("Welcome")

@app.get('/test', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input name="text" type="text" value="Text to detetct AEs" />
         <input type="submit" />'''

@app.post('/test') #prediction on data
def test(text:str = Form(...)): #input is from forms
    result= testmodel(text)
    
    return { #returning a dictionary as endpoint
        "ACTUALL SENTENCE": text,
        "RESULT":result
    }