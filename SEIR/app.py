import dash
import dash_core_components as dcc 
import dash_html_components as html
import sys
from dash.dependencies import Input, Output

from parameters import DiseaseParams, SimOpts, PlotOpts, Country_Info
from SEIR_Sim import Store_Results, SEIR_Model


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://www.w3schools.com/w3css/4/w3.css']

print(external_stylesheets)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

disease_parameters = DiseaseParams()
simulation_paramters = SimOpts()
country_parameters = Country_Info()

def custom_slider(title: str, id_in: str, min_val: int, max_val: int, initial_value: int, step_size):
    return html.Div(children=[html.P(className='w3-justify', children=[title]), 
                            dcc.Slider(
                                id=id_in,
                                min=min_val,
                                max=max_val,
                                value=initial_value,
                                marks={
                                    min_val: str(min_val),
                                    max_val: str(max_val)
                                },
                                step=step_size,
                                tooltip={
                                    'always_visible': True,
                                    'placement': 'topLeft'
                                }
                            )
    ])

app.layout = html.Div([
    html.Div(className = "w3-top", children = [
        html.Div(className="w3-bar w3-black w3-card w3-padding-large", children='Demo')
    ]
    ),
    html.Div(className='w3-content w3-margin-left w3-margin-right', children=[
        html.H1(className='w3-wide w3-center', children = ['SEIR Model']),
        html.P(className='w3-justify w3-center', children=['A very simple app to play around with the SEIR model.']),
        html.Div([
            html.Div(className='w3-col m2', children=[
                custom_slider('Population', 'pop_slider', 0, 10**7, 10**6, 1000),
                custom_slider('Simulation time (days)', 'days_slider', 0, 365, 200, 1),
                custom_slider('Beta (rate S-> E)', 'beta_slider', 0, 5, 1/2.5, 0.1),
                custom_slider('Gamma (rate I -> R)', 'gamma_slider', 0, 2, 1/13., 0.1),
                custom_slider('Sigma (rate E -> I)', 'sigma_slider', 0, 4, 1/2., 0.1),
                html.P(className='w3-justify', children='Lockdown'),
                dcc.Dropdown(id='lockdown',
                            className='w3-justify',
                            options = [
                                {'label':'Yes', 'value': 1},
                                {'label':'No', 'value': 0}
                            ],
                            value= 0,
                            #labelStyle={'display':'inline-block'}
                            ),
                custom_slider('Lockdown #days delay', 'lockdown_days', 0, 100, 25, 1),
                custom_slider('Intially exposed persons', 'init_expos', 1, 100, 1, 1),
                
            ]),
            html.Div(className='w3-col m10', children=[
                dcc.Graph(
                    id = 'SEIR-graph'
                )
            ])
        ])
    ],style={'margin-top': '70px', 'max-width':'2000px'})            
])


@app.callback(Output('SEIR-graph','figure'),
            [Input('pop_slider','value'),
             Input('days_slider','value'),
             Input('beta_slider','value'),
             Input('gamma_slider','value'),
             Input('sigma_slider','value'),
             Input('lockdown','value'),
             Input('lockdown_days','value'),
             Input('init_expos','value')
            ]

)
def calc_SEIR_graph(slider_pop,slider_day,beta_slider,gamma_slider,sigma_slider,lockdown,lockdown_days,init_expos):
    print(slider_pop)
    country_parameters.country_population = slider_pop
    simulation_paramters.sim_length = slider_day
    disease_parameters.beta_init = beta_slider
    disease_parameters.gamma = gamma_slider
    disease_parameters.sigma = sigma_slider
    simulation_paramters.lockdown=bool(lockdown)
    simulation_paramters.lockdown_delay=lockdown_days
    simulation_paramters.initial_exposed=init_expos
    print(country_parameters.country_population)

    model = SEIR_Model(country_parameters.country_name, country_parameters.country_population)
    results = model.run_seir(disease_parameters, simulation_paramters)
    return {
        'data': [
            {'x': results.T, 'y': results.S, 'name': 'Susceptible'},
            {'x': results.T, 'y': results.E, 'name': 'Exposed'},
            {'x': results.T, 'y': results.I, 'name': 'Infected'},
            {'x': results.T, 'y': results.R, 'name': 'Recovered'},
            {'x': results.T, 'y': results.D, 'name': 'Deaths'},
        ],
        'layout': {
            'xaxis': {'showgrid': False},
            'yaxis': {'showgrid': False},
            'legend': {'orientation':'h'},
            'height': '800'
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True)
