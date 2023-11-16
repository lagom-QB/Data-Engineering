from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import uvicorn
from pydantic import BaseModel

import os
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from src.transform_data import transform_data_into_features_and_targets
from src.model_prediction import predict_res

app_home = FastAPI()
templates = Jinja2Templates(directory="templates")
cols: list[str] = ['Wk', 'Numeric_Day', 'Numeric_Home', 'Numeric_Away', 'Numeric_Time',
        'xGHome_xGAway_1', 'xGHome_xGAway_2', 'xGHome_xGAway_3']

class TeamStats(BaseModel):
    Wk: int
    Day: str
    Time: Optional[str] = None
    Home: str
    Away: str


@app_home.get("/")
async def root(stats: TeamStats) -> HTMLResponse:
    """
    Returns a base page to input the team stats
    """
    return templates.TemplateResponse("index.html", {"request": Request, "stats": stats}) # type: ignore

@app_home.get("/submit")
async def submit(stats: TeamStats) -> Dict[str, str]:
    """
    Returns a base page to input the team stats
    """
    return {'Week': stats.Wk, 'Day': stats.Day, 'Time': stats.Time, 'Home': stats.Home, 'Away': stats.Away} # type: ignore

@app_home.post("/predict", response_model=Dict[str, float])  # Specify the response model type
async def predict(data:TeamStats) -> str:
    """
    Predicts the score of a match based on the team stats
    data: TeamStatistics expected
    model_type: home or away
        returns: Dict[str, float] with the predicted score
    """
    # Load the inputs into a dataframe
    data = pd.DataFrame(columns=['Wk', 'Day', 'Time', 'Home', 'Away']) # type: ignore
    data.loc[0] = [Wk, Day, Time, Home, Away] # type: ignore

    features_home, target_home = transform_data_into_features_and_targets(df=data, score='ScoreHome')
    features_away, target_away = transform_data_into_features_and_targets(df=data, score='ScoreAway')

    prediction: str = predict_res(features_home) # type: ignore

    return prediction

if __name__ == '__main__':
    # What is the current working directory?
    print(os.getcwd())
    uvicorn.run(app_home, host='0.0.0.0', port=8000)